"""DLLM Block-wise Mask 实现方案对比

对比三种实现方式：
1. Cascade Attention (SGLang 方式): Ragged + Paged + merge_state
   - Ragged: 当前 block 内部双向 attention (causal=False)
   - Paged: 访问之前所有 cached blocks (causal=False)
   - merge_state: 合并两个阶段的 softmax 状态

2. BatchPrefillWithRaggedKVCacheWrapper(block_extend=True, q_offsets=...) (FlashInfer 优化方式)
   - 单次 kernel launch
   - tile 级别跳过无效计算

3. single_prefill_with_kv_cache 串行方式 (参考基准)
   - 每个 chunk 独立调用
   - tile 级别跳过

Mask 规则: mask[q, k] = (q_global // B) >= (k_global // B)
"""

import torch
import time
import math
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state,
    single_prefill_with_kv_cache,
)


def compute_block_extend_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    q_offset: int = 0,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    使用 custom_mask 计算 Block Extend Attention 的参考结果

    Mask 规则: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)

    Args:
        q: [qo_len, num_heads, head_dim]
        k: [kv_len, num_kv_heads, head_dim]
        v: [kv_len, num_kv_heads, head_dim]
        block_size: DLLM block 大小
        q_offset: Q 的全局起始位置
        sm_scale: softmax scale

    Returns:
        output: [qo_len, num_heads, head_dim]
    """
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    head_dim = q.shape[-1]
    device = q.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # 构造 custom_mask
    # q_global = q_local + q_offset
    # mask[q, k] = (q_global // B) >= (k // B)
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    q_block = q_pos.unsqueeze(1) // block_size  # [qo_len, 1]
    k_block = k_pos.unsqueeze(0) // block_size  # [1, kv_len]
    mask_2d = (q_block >= k_block).to(torch.uint8)  # [qo_len, kv_len]

    return single_prefill_with_kv_cache(
        q,
        k,
        v,
        custom_mask=mask_2d,
        sm_scale=sm_scale,
    )


def test_incremental_batchprefill_step_by_step_with_cuda_graph(
    num_requests: int = 4,
    tokens_per_request: int = 256,
    block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
    backend: str = "fa2",
):
    """
    真实模拟 DLLM 增量 Prefill 的分步执行流程 + CUDA Graph

    关键点:
      1. 必须分步执行，每个 chunk step 依赖前一个 step 的 KV cache
      2. SGLang Cascade 强制 chunk_size = block_size
      3. BatchPrefillWithRaggedKVCacheWrapper(block_extend=True) 可以使用更大的 chunk_size，减少 step 数
      4. 启用 CUDA Graph 减少 CPU 开销和 kernel launch 延迟

    CUDA Graph 注意事项:
      - plan() 包含 CPU-GPU 同步，不能在 capture 期间执行
      - 必须为每个 step 创建独立的 wrapper，在 capture 前完成 plan
      - 只 capture run() 操作
    """
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]

    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim**0.5)

    # Baseline chunk_size = block_size
    baseline_chunk_size = block_size
    baseline_num_chunks = tokens_per_request // baseline_chunk_size

    print(f"\n{'=' * 80}")
    print("真实分步执行 + CUDA Graph: DLLM 增量 Prefill 性能对比")
    print(f"{'=' * 80}")
    print("配置:")
    print(f"  num_requests        = {num_requests}")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  block_size     = {block_size}")
    print(f"  chunk_sizes to test = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print("\n关键特性:")
    print("  - 分步执行: 每个 step 必须等待前一个 step 完成")
    print(f"  - SGLang 强制 chunk_size = block_size = {block_size}")
    print(
        "  - BatchPrefillWithRaggedKVCacheWrapper(block_extend=True) 可以使用更大的 chunk_size"
    )
    print("  - CUDA Graph: 减少 CPU 开销和 kernel launch 延迟")

    # 数据准备，生成每个 request 的完整 Q, K, V
    all_qs = [
        torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(num_requests)
    ]
    all_ks = [
        torch.randn(
            tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        for _ in range(num_requests)
    ]
    all_vs = [
        torch.randn(
            tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        for _ in range(num_requests)
    ]

    # 为每个 request 分 chunk
    def split_chunks(tensor, chunk_size):
        return [
            tensor[i * chunk_size : (i + 1) * chunk_size]
            for i in range(tensor.shape[0] // chunk_size)
        ]

    qs_chunks = [split_chunks(q, baseline_chunk_size) for q in all_qs]
    ks_chunks = [split_chunks(k, baseline_chunk_size) for k in all_ks]
    vs_chunks = [split_chunks(v, baseline_chunk_size) for v in all_vs]

    results = {}

    # 正确性验证 (抽样验证第一个 request)
    print(f"\n{'=' * 80}")
    print("正确性验证 (抽样验证 request_0)")
    print(f"{'=' * 80}")
    print("  参考实现: single_prefill_with_kv_cache + custom_mask")
    print("  Mask 规则: mask[q,k] = ((q + offset) // B) >= (k // B)")

    # 抽样验证第一个 request
    req_idx = 0
    k_req = all_ks[req_idx]
    v_req = all_vs[req_idx]

    # 累积 KV buffer
    k_cumul_verify = [
        k_req[: (i + 1) * baseline_chunk_size] for i in range(baseline_num_chunks)
    ]
    v_cumul_verify = [
        v_req[: (i + 1) * baseline_chunk_size] for i in range(baseline_num_chunks)
    ]

    # 计算参考结果
    ref_outputs = []
    for step_idx in range(baseline_num_chunks):
        q_offset = step_idx * baseline_chunk_size
        ref_out = compute_block_extend_reference(
            qs_chunks[req_idx][step_idx],
            k_cumul_verify[step_idx],
            v_cumul_verify[step_idx],
            block_size=block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
        )
        ref_outputs.append(ref_out)

    bbe_outputs = []
    workspace_verify = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_indptr_verify = torch.tensor(
        [0, baseline_chunk_size], dtype=torch.int32, device=device
    )
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        q_offset_tensor = torch.tensor(
            [step_idx * baseline_chunk_size], dtype=torch.int32, device=device
        )

        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace_verify,
            kv_layout="NHD",
            block_extend=True,
            block_size=block_size,
            backend=backend,
        )
        wrapper.plan(
            qo_indptr=qo_indptr_verify,
            kv_indptr=kv_indptr,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_data_type=dtype,
            sm_scale=sm_scale,
            causal=False,
            q_offsets=q_offset_tensor,
        )
        bbe_out = wrapper.run(
            qs_chunks[req_idx][step_idx],
            k_cumul_verify[step_idx],
            v_cumul_verify[step_idx],
        )
        bbe_outputs.append(bbe_out)

    # 验证
    tol = 1e-2
    bbe_max_diff = max(
        (bbe_outputs[i] - ref_outputs[i]).abs().max().item()
        for i in range(baseline_num_chunks)
    )
    bbe_pass = bbe_max_diff < tol
    print("\n  BatchPrefillWithRaggedKVCacheWrapper(block_extend=True):")
    print(f"        max_diff = {bbe_max_diff:.6f}, tolerance = {tol}")
    print(f"        {' PASS' if bbe_pass else ' FAIL'}")

    if not bbe_pass:
        print("\n   正确性验证未通过，但继续执行性能测试")
    else:
        print(
            "\n   BatchPrefillWithRaggedKVCacheWrapper(block_extend=True) 正确性验证通过"
        )

    del workspace_verify, bbe_outputs, ref_outputs, k_cumul_verify, v_cumul_verify
    torch.cuda.empty_cache()

    # Baseline 2: Custom Mask (BatchPrefillWithRaggedKVCacheWrapper + custom_mask) + CUDA Graph
    print(f"\n{'=' * 80}")
    print("[Baseline 2] Custom Mask (BatchPrefill + custom_mask)")
    print(f"{'=' * 80}")
    print(f"  num_steps: {baseline_num_chunks}")
    print(f"  num_requests: {num_requests}")
    print("  每个 step: BatchPrefillWithRaggedKVCacheWrapper + custom_mask")
    print("  每个 step 内 kernel: 1 次 (batch 处理所有 requests)")

    # 预分配 Q buffers (concat 所有 requests)
    cm_q_buffers = []
    for step_idx in range(baseline_num_chunks):
        q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        cm_q_buffers.append(torch.cat(q_list, dim=0))

    # 预分配累积 KV buffers (concat 所有 requests)
    cm_k_buffers = []
    cm_v_buffers = []
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
        v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
        cm_k_buffers.append(torch.cat(k_cumul_list, dim=0))
        cm_v_buffers.append(torch.cat(v_cumul_list, dim=0))

    # 构造 flattened custom_mask (batch 版本)
    # custom_mask shape: (sum(q_len[i] * k_len[i] for i in range(batch_size)))
    # 每个 request 的 mask 都相同，concat num_requests 次
    custom_mask_buffers = []
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        q_offset = step_idx * baseline_chunk_size
        # 构造单个 request 的 2D mask: [q_len, kv_len]
        q_pos = torch.arange(baseline_chunk_size, device=device) + q_offset
        k_pos = torch.arange(kv_len, device=device)
        q_block = q_pos.unsqueeze(1) // block_size
        k_block = k_pos.unsqueeze(0) // block_size
        mask_2d = q_block >= k_block  # [q_len, kv_len], bool
        # flatten 并 repeat 为 batch
        mask_flat = mask_2d.flatten()  # [q_len * kv_len]
        # 所有 requests 的 mask 相同，concat
        batch_mask = mask_flat.repeat(num_requests)  # [num_requests * q_len * kv_len]
        custom_mask_buffers.append(batch_mask)

    # qo_indptr 和 kv_indptr
    cm_qo_indptr = torch.tensor(
        [i * baseline_chunk_size for i in range(num_requests + 1)],
        dtype=torch.int32,
        device=device,
    )
    cm_kv_indptr_list = []
    for step_idx in range(baseline_num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        cm_kv_indptr_list.append(
            torch.tensor(
                [i * kv_len for i in range(num_requests + 1)],
                dtype=torch.int32,
                device=device,
            )
        )

    # 为每个 step 创建独立的 wrapper 并完成 plan
    # 注意: custom_mask 只在 FA2 后端支持，FA3 不支持，强制使用 fa2
    print("  创建 wrappers 并完成 plan...")
    cm_wrappers = []
    for step_idx in range(baseline_num_chunks):
        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend="fa2",
        )
        wrapper.plan(
            qo_indptr=cm_qo_indptr,
            kv_indptr=cm_kv_indptr_list[step_idx],
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            custom_mask=custom_mask_buffers[step_idx],
            causal=False,
            sm_scale=sm_scale,
        )
        cm_wrappers.append(wrapper)

    cm_output = torch.empty(
        num_requests * baseline_chunk_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    def run_custom_mask_pipeline():
        for step_idx in range(baseline_num_chunks):
            cm_output.copy_(
                cm_wrappers[step_idx].run(  # noqa: F821
                    cm_q_buffers[step_idx],
                    cm_k_buffers[step_idx],
                    cm_v_buffers[step_idx],
                )
            )

    if verbose:
        print("  Step 流程预览:")
        for step_id in range(baseline_num_chunks):
            kv_len = (step_id + 1) * baseline_chunk_size
            print(
                f"    Step {step_id}: Q[{step_id * baseline_chunk_size}:{(step_id + 1) * baseline_chunk_size}] attend to KV[0:{kv_len}]"
            )

    # Warmup
    print("  Warmup...")
    for _ in range(warmup_iters):
        run_custom_mask_pipeline()
    torch.cuda.synchronize()

    # CUDA Graph capture
    print("  Capturing CUDA Graph...")
    cm_stream = torch.cuda.Stream()
    with torch.cuda.stream(cm_stream):
        run_custom_mask_pipeline()
    cm_stream.synchronize()

    cm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cm_graph, stream=cm_stream):
        run_custom_mask_pipeline()

    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cm_graph.replay()
    torch.cuda.synchronize()

    # Benchmark
    print("  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cm_graph.replay()
    torch.cuda.synchronize()
    cm_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

    results["custom_mask_baseline"] = {
        "time_cuda_graph_ms": cm_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": baseline_num_chunks,
        "method": "Custom Mask (Baseline 2)",
    }
    print(
        f"  => cuda_graph: {cm_cuda_graph_time:.3f} ms ({cm_cuda_graph_time / baseline_num_chunks:.3f} ms/step × {baseline_num_chunks} steps)"
    )

    del cm_graph, custom_mask_buffers, cm_wrappers
    torch.cuda.empty_cache()

    # Baseline 1: SGLang Cascade (chunk_size = block_size) + CUDA Graph
    print(f"\n{'=' * 80}")
    print(
        f"[Baseline 1] SGLang Cascade (chunk_size = block_size = {baseline_chunk_size})"
    )
    print(f"{'=' * 80}")
    print(f"  num_steps: {baseline_num_chunks}")
    print("  每个 step: BatchRagged(当前chunk) + BatchRagged(prefix) + merge_state")
    print("  每个 step 内 kernel: 2-3 次")

    # 预分配所有 buffer（用于 CUDA Graph）
    q_current_buffers = []
    k_current_buffers = []
    v_current_buffers = []
    for step_idx in range(baseline_num_chunks):
        q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        k_list = [ks_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        v_list = [vs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
        q_current_buffers.append(torch.cat(q_list, dim=0))
        k_current_buffers.append(torch.cat(k_list, dim=0))
        v_current_buffers.append(torch.cat(v_list, dim=0))

    # 预分配 prefix KV buffer
    k_prefix_buffers = [None]
    v_prefix_buffers = [None]
    for step_idx in range(1, baseline_num_chunks):
        prefix_len = step_idx * baseline_chunk_size
        k_prefix_list = [
            all_ks[req_idx][:prefix_len] for req_idx in range(num_requests)
        ]
        v_prefix_list = [
            all_vs[req_idx][:prefix_len] for req_idx in range(num_requests)
        ]
        k_prefix_buffers.append(torch.cat(k_prefix_list, dim=0))
        v_prefix_buffers.append(torch.cat(v_prefix_list, dim=0))

    cascade_output = torch.empty(
        num_requests * baseline_chunk_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    # indptr
    qo_indptr_current = torch.tensor(
        [i * baseline_chunk_size for i in range(num_requests + 1)],
        dtype=torch.int32,
        device=device,
    )
    kv_indptr_prefix_list = [None]
    for step_idx in range(1, baseline_num_chunks):
        prefix_len = step_idx * baseline_chunk_size
        kv_indptr_prefix_list.append(
            torch.tensor(
                [i * prefix_len for i in range(num_requests + 1)],
                dtype=torch.int32,
                device=device,
            )
        )

    # 为每个 step 创建独立的 wrapper 并完成 plan（关键！）
    print("  创建 wrappers 并完成 plan...")
    cascade_wrappers_current = []
    cascade_wrappers_prefix = []

    for step_idx in range(baseline_num_chunks):
        # 当前 chunk 的 wrapper
        wrapper_current = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            backend=backend,
        )
        wrapper_current.plan(
            qo_indptr=qo_indptr_current,
            kv_indptr=qo_indptr_current,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=False,
            sm_scale=sm_scale,
        )
        cascade_wrappers_current.append(wrapper_current)

        # prefix 的 wrapper（step 0 没有 prefix）
        if step_idx == 0:
            cascade_wrappers_prefix.append(None)
        else:
            wrapper_prefix = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                backend=backend,
            )
            wrapper_prefix.plan(
                qo_indptr=qo_indptr_current,
                kv_indptr=kv_indptr_prefix_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=False,
                sm_scale=sm_scale,
            )
            cascade_wrappers_prefix.append(wrapper_prefix)

    o1_buffer = torch.empty(
        num_requests * baseline_chunk_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    s1_buffer = torch.empty(
        num_requests * baseline_chunk_size,
        num_heads,
        dtype=torch.float32,
        device=device,
    )
    o2_buffer = torch.empty(
        num_requests * baseline_chunk_size,
        num_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    s2_buffer = torch.empty(
        num_requests * baseline_chunk_size,
        num_heads,
        dtype=torch.float32,
        device=device,
    )

    def run_cascade_pipeline():
        for step_idx in range(baseline_num_chunks):
            q_batch = q_current_buffers[step_idx]
            k_current = k_current_buffers[step_idx]
            v_current = v_current_buffers[step_idx]

            if step_idx == 0:
                cascade_output.copy_(
                    cascade_wrappers_current[step_idx].run(  # noqa: F821
                        q_batch, k_current, v_current
                    )
                )
            else:
                o1, s1 = cascade_wrappers_current[step_idx].run_return_lse(  # noqa: F821
                    q_batch, k_current, v_current
                )
                o1_buffer.copy_(o1)
                s1_buffer.copy_(s1)

                o2, s2 = cascade_wrappers_prefix[step_idx].run_return_lse(  # noqa: F821
                    q_batch, k_prefix_buffers[step_idx], v_prefix_buffers[step_idx]
                )
                o2_buffer.copy_(o2)
                s2_buffer.copy_(s2)

                o, _ = merge_state(o1_buffer, s1_buffer, o2_buffer, s2_buffer)
                cascade_output.copy_(o)

    # 显示 step 流程
    if verbose:
        print("  Step 流程预览:")
        for step_id in range(baseline_num_chunks):
            prefix_len = step_id * baseline_chunk_size
            if step_id == 0:
                print(
                    f"    Step {step_id}: current_chunk[{baseline_chunk_size}] only (no prefix)"
                )
            else:
                print(
                    f"    Step {step_id}: current_chunk[{baseline_chunk_size}] + prefix[{prefix_len}] + merge"
                )

    # Warmup
    print("  Warmup...")
    for _ in range(warmup_iters):
        run_cascade_pipeline()
    torch.cuda.synchronize()

    # CUDA Graph capture
    print("  Capturing CUDA Graph...")
    cascade_stream = torch.cuda.Stream()
    with torch.cuda.stream(cascade_stream):
        run_cascade_pipeline()
    cascade_stream.synchronize()

    cascade_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cascade_graph, stream=cascade_stream):
        run_cascade_pipeline()

    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()

    # Benchmark
    print("  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()
    cascade_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

    results["cascade_baseline"] = {
        "time_cuda_graph_ms": cascade_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": baseline_num_chunks,
        "method": "SGLang Cascade (Baseline 1)",
    }
    print(
        f"  => cuda_graph: {cascade_cuda_graph_time:.3f} ms ({cascade_cuda_graph_time / baseline_num_chunks:.3f} ms/step × {baseline_num_chunks} steps)"
    )

    del cascade_wrappers_current, cascade_wrappers_prefix, cascade_graph
    torch.cuda.empty_cache()

    # 对比方案: BatchPrefillWithRaggedKVCacheWrapper(block_extend=True) (不同 chunk_size) + CUDA Graph

    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(
                f"\n[跳过] chunk_size={test_chunk_size} 无法整除 tokens_per_request={tokens_per_request}"
            )
            continue

        num_chunks_bbe = tokens_per_request // test_chunk_size

        print(f"\n{'-' * 60}")
        print(
            f"[BatchPrefillWithRaggedKVCacheWrapper] chunk_size = {test_chunk_size} (block_extend=True)"
        )
        print(f"{'-' * 60}")
        print(
            f"  num_steps: {num_chunks_bbe} (比 Baseline 少 {baseline_num_chunks - num_chunks_bbe} 步)"
        )
        print("  每个 step 内 kernel: 1 次")

        # 为每个 request 分 chunk
        qs_chunks_bbe = [split_chunks(q, test_chunk_size) for q in all_qs]

        # 预分配所有 buffer
        bbe_q_buffers = []
        for step_idx in range(num_chunks_bbe):
            q_list = [
                qs_chunks_bbe[req_idx][step_idx] for req_idx in range(num_requests)
            ]
            bbe_q_buffers.append(torch.cat(q_list, dim=0))

        bbe_k_buffers = []
        bbe_v_buffers = []
        for step_idx in range(num_chunks_bbe):
            kv_len = (step_idx + 1) * test_chunk_size
            k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
            v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
            bbe_k_buffers.append(torch.cat(k_cumul_list, dim=0))
            bbe_v_buffers.append(torch.cat(v_cumul_list, dim=0))

        bbe_qo_indptr = torch.tensor(
            [i * test_chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32,
            device=device,
        )
        bbe_kv_indptr_list = []
        bbe_q_offsets_list = []
        for step_idx in range(num_chunks_bbe):
            kv_len = (step_idx + 1) * test_chunk_size
            bbe_kv_indptr_list.append(
                torch.tensor(
                    [i * kv_len for i in range(num_requests + 1)],
                    dtype=torch.int32,
                    device=device,
                )
            )
            q_offset = step_idx * test_chunk_size
            """
            bbe_q_offsets_list = [
                # step_idx=0: q_offset = 0 * 64 = 0
                tensor([0, 0, 0, 0], dtype=int32),     # shape: (4,)

                # step_idx=1: q_offset = 1 * 64 = 64
                tensor([64, 64, 64, 64], dtype=int32), # shape: (4,)

                # step_idx=2: q_offset = 2 * 64 = 128
                tensor([128, 128, 128, 128], dtype=int32), # shape: (4,)
            ]
            """
            bbe_q_offsets_list.append(
                torch.full((num_requests,), q_offset, dtype=torch.int32, device=device)
            )

        bbe_output = torch.empty(
            num_requests * test_chunk_size,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        print("  创建 wrappers 并完成 plan...")
        bbe_wrappers = []
        for step_idx in range(num_chunks_bbe):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend=backend,
            )
            wrapper.plan(
                qo_indptr=bbe_qo_indptr,
                kv_indptr=bbe_kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=bbe_q_offsets_list[step_idx],
            )
            bbe_wrappers.append(wrapper)

        def run_bbe_pipeline():
            for step_idx in range(num_chunks_bbe):
                bbe_output.copy_(
                    bbe_wrappers[step_idx].run(  # noqa: F821
                        bbe_q_buffers[step_idx],
                        bbe_k_buffers[step_idx],
                        bbe_v_buffers[step_idx],
                    )
                )

        # 显示 step 流程
        if verbose:
            print("  Step 流程预览:")
            for step_id in range(num_chunks_bbe):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(
                    f"    Step {step_id}: Q[{q_offset}:{q_offset + test_chunk_size}] attend to KV[0:{kv_len}] (q_offset={q_offset})"
                )

        # Warmup
        print("  Warmup...")
        for _ in range(warmup_iters):
            run_bbe_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        print("  Capturing CUDA Graph...")
        bbe_stream = torch.cuda.Stream()
        with torch.cuda.stream(bbe_stream):
            run_bbe_pipeline()
        bbe_stream.synchronize()

        bbe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bbe_graph, stream=bbe_stream):
            run_bbe_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()

        # Benchmark
        print("  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()
        bbe_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

        results[f"ragged_prefill_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": bbe_cuda_graph_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_chunks_bbe,
            "method": "BatchPrefillWithRaggedKVCacheWrapper (block_extend)",
        }
        print(
            f"  => cuda_graph: {bbe_cuda_graph_time:.3f} ms ({bbe_cuda_graph_time / num_chunks_bbe:.3f} ms/step × {num_chunks_bbe} steps)"
        )

        del bbe_wrappers, bbe_graph
        torch.cuda.empty_cache()

    # Custom Mask 不同 chunk_size 测试 (不要求 chunk_size = block_size)
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(
                f"\n[跳过] chunk_size={test_chunk_size} 无法整除 tokens_per_request={tokens_per_request}"
            )
            continue

        # 跳过已经测试过的 baseline chunk_size
        if test_chunk_size == baseline_chunk_size:
            continue

        num_chunks_cm = tokens_per_request // test_chunk_size

        print(f"\n{'-' * 60}")
        print(f"[Custom Mask] chunk_size = {test_chunk_size}")
        print(f"{'-' * 60}")
        print(
            f"  num_steps: {num_chunks_cm} (比 Baseline 少 {baseline_num_chunks - num_chunks_cm} 步)"
        )
        print("  每个 step 内 kernel: 1 次 (batch 处理所有 requests)")

        # 为每个 request 分 chunk
        qs_chunks_cm = [split_chunks(q, test_chunk_size) for q in all_qs]

        # 预分配 Q buffers (concat 所有 requests)
        cm_q_buffers_var = []
        for step_idx in range(num_chunks_cm):
            q_list = [
                qs_chunks_cm[req_idx][step_idx] for req_idx in range(num_requests)
            ]
            cm_q_buffers_var.append(torch.cat(q_list, dim=0))

        # 预分配累积 KV buffers (concat 所有 requests)
        cm_k_buffers_var = []
        cm_v_buffers_var = []
        for step_idx in range(num_chunks_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
            v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
            cm_k_buffers_var.append(torch.cat(k_cumul_list, dim=0))
            cm_v_buffers_var.append(torch.cat(v_cumul_list, dim=0))

        # 构造 flattened custom_mask (batch 版本)
        # DLLM blockwise mask: mask[q, k] = ((q + q_offset) // B) >= (k // B)
        cm_mask_buffers_var = []
        for step_idx in range(num_chunks_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            q_offset = step_idx * test_chunk_size
            # 构造单个 request 的 2D mask: [q_len, kv_len]
            q_pos = torch.arange(test_chunk_size, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // block_size
            k_block = k_pos.unsqueeze(0) // block_size
            mask_2d = q_block >= k_block  # [q_len, kv_len], bool
            # flatten 并 repeat 为 batch
            mask_flat = mask_2d.flatten()
            batch_mask = mask_flat.repeat(num_requests)
            cm_mask_buffers_var.append(batch_mask)

        # qo_indptr 和 kv_indptr
        cm_qo_indptr_var = torch.tensor(
            [i * test_chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32,
            device=device,
        )
        cm_kv_indptr_list_var = []
        for step_idx in range(num_chunks_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            cm_kv_indptr_list_var.append(
                torch.tensor(
                    [i * kv_len for i in range(num_requests + 1)],
                    dtype=torch.int32,
                    device=device,
                )
            )

        # 为每个 step 创建独立的 wrapper 并完成 plan
        print("  创建 wrappers 并完成 plan...")
        cm_wrappers_var = []
        for step_idx in range(num_chunks_cm):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                backend="fa2",
            )
            wrapper.plan(
                qo_indptr=cm_qo_indptr_var,
                kv_indptr=cm_kv_indptr_list_var[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                custom_mask=cm_mask_buffers_var[step_idx],
                causal=False,
                sm_scale=sm_scale,
            )
            cm_wrappers_var.append(wrapper)

        cm_output_var = torch.empty(
            num_requests * test_chunk_size,
            num_heads,
            head_dim,
            dtype=dtype,
            device=device,
        )

        def run_cm_pipeline_var():
            for step_idx in range(num_chunks_cm):
                cm_output_var.copy_(
                    cm_wrappers_var[step_idx].run(  # noqa: F821
                        cm_q_buffers_var[step_idx],
                        cm_k_buffers_var[step_idx],
                        cm_v_buffers_var[step_idx],
                    )
                )

        if verbose:
            print("  Step 流程预览:")
            for step_id in range(num_chunks_cm):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(
                    f"    Step {step_id}: Q[{q_offset}:{q_offset + test_chunk_size}] attend to KV[0:{kv_len}]"
                )

        # Warmup
        print("  Warmup...")
        for _ in range(warmup_iters):
            run_cm_pipeline_var()
        torch.cuda.synchronize()

        # CUDA Graph capture
        print("  Capturing CUDA Graph...")
        cm_stream_var = torch.cuda.Stream()
        with torch.cuda.stream(cm_stream_var):
            run_cm_pipeline_var()
        cm_stream_var.synchronize()

        cm_graph_var = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cm_graph_var, stream=cm_stream_var):
            run_cm_pipeline_var()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()

        # Benchmark
        print("  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()
        cm_var_time = (time.perf_counter() - start) / bench_iters * 1000

        results[f"cm_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": cm_var_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_chunks_cm,
            "method": "Custom Mask",
        }
        print(
            f"  => cuda_graph: {cm_var_time:.3f} ms ({cm_var_time / num_chunks_cm:.3f} ms/step × {num_chunks_cm} steps)"
        )

        del cm_graph_var, cm_mask_buffers_var, cm_wrappers_var
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print("结果汇总 (分步执行 + CUDA Graph)")
    print(f"{'=' * 80}")

    cm_baseline_time = results["custom_mask_baseline"]["time_cuda_graph_ms"]
    cascade_baseline_time = results["cascade_baseline"]["time_cuda_graph_ms"]

    print("\n说明:")
    print("  - Baseline 1: SGLang Cascade (BatchPrefill + merge_state)")
    print("  - Baseline 2: Custom Mask (BatchPrefill + custom_mask)")
    print("  - vs Base1: 相对于 SGLang Cascade 的加速比")
    print("  - vs Base2: 相对于 Custom Mask 的加速比")

    print(
        f"\n{'方案':<40} | {'chunk':>6} | {'steps':>6} | {'cuda_graph(ms)':>10} | {'ms/step':>10} | {'vs Base1':>10} | {'vs Base2':>10}"
    )
    print(
        f"{'-' * 40}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}"
    )

    # Baseline 1: SGLang Cascade
    r = results["cascade_baseline"]
    print(
        f"{'[Baseline 1] SGLang Cascade':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {'1.00x':>10} | {cm_baseline_time / cascade_baseline_time:>9.2f}x"
    )

    # Baseline 2: Custom Mask
    r = results["custom_mask_baseline"]
    print(
        f"{'[Baseline 2] Custom Mask':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {cascade_baseline_time / cm_baseline_time:>9.2f}x | {'1.00x':>10}"
    )

    cm_keys = [k for k in results.keys() if k.startswith("cm_chunk")]
    cm_keys_sorted = sorted(cm_keys, key=lambda k: results[k]["chunk_size"])
    for key in cm_keys_sorted:
        r = results[key]
        speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        print(
            f"{'Custom Mask':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {speedup_vs_cascade:>9.2f}x | {speedup_vs_cm:>9.2f}x"
        )

    ragged_keys = [k for k in results.keys() if k.startswith("ragged_prefill_chunk")]
    ragged_keys_sorted = sorted(ragged_keys, key=lambda k: results[k]["chunk_size"])
    for key in ragged_keys_sorted:
        r = results[key]
        speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        print(
            f"{'BatchPrefillWithRaggedKVCacheWrapper (block_extend)':<40} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {speedup_vs_cascade:>9.2f}x | {speedup_vs_cm:>9.2f}x"
        )

    return results


def test_incremental_singlereq_prefill_step_by_step_with_cuda_graph(
    tokens_per_request: int = 512,
    block_size: int = 32,
    chunk_sizes: list = None,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
    backend: str = "fa2",
):
    """
    单请求增量 Prefill 分步执行测试 + CUDA Graph

    场景说明:
      - 单个 request 的增量 prefill
      - 必须分步执行: chunk0 完成后才能算 chunk1 (流水线依赖)
      - SGLang 即使 batch_size=1 也使用 BatchPrefill 接口

    Baseline (SGLang 3阶段 DLLM Cascade):
      1. BatchPrefillWithRaggedKVCacheWrapper: 当前 chunk (causal=False)
      2. BatchPrefillWithRaggedKVCacheWrapper: prefix KV (causal=False)
      3. merge_state: 合并两部分结果
      每个 step: 2-3 kernel launches

    对比方案:
      1. single_prefill_with_kv_cache + CUDA Graph
      2. BatchPrefillWithRaggedKVCacheWrapper + CUDA Graph
      每个 step: 1 kernel launch
    """
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]

    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim**0.5)

    # Baseline: chunk_size = block_size (SGLang 约束)
    baseline_chunk_size = block_size
    num_chunks = tokens_per_request // baseline_chunk_size

    print(f"\n{'=' * 80}")
    print("单请求增量 Prefill 分步执行 + CUDA Graph")
    print(f"{'=' * 80}")
    print("配置:")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  block_size     = {block_size}")
    print(f"  chunk_sizes to test = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print("\n场景说明:")
    print("  - 单请求 (batch_size=1)，但使用 BatchPrefill 接口 (SGLang 做法)")
    print("  - 分步执行: chunk0 完成后才能算 chunk1 (流水线依赖)")
    print(f"  - num_steps = {num_chunks}")

    # 数据准备
    # 单个 request 的完整 Q, K, V
    q_full = torch.randn(
        tokens_per_request, num_heads, head_dim, dtype=dtype, device=device
    )
    k_full = torch.randn(
        tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_full = torch.randn(
        tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    # 分 chunk
    def split_chunks(tensor, chunk_size):
        return [
            tensor[i * chunk_size : (i + 1) * chunk_size]
            for i in range(tensor.shape[0] // chunk_size)
        ]

    qs_chunks = split_chunks(q_full, baseline_chunk_size)
    ks_chunks = split_chunks(k_full, baseline_chunk_size)
    vs_chunks = split_chunks(v_full, baseline_chunk_size)

    # 累积 KV buffer
    k_cumul_list = [
        torch.cat([k_full[: (i + 1) * baseline_chunk_size]], dim=0)
        for i in range(num_chunks)
    ]
    v_cumul_list = [
        torch.cat([v_full[: (i + 1) * baseline_chunk_size]], dim=0)
        for i in range(num_chunks)
    ]

    results = {}

    # 正确性验证
    print(f"\n{'=' * 80}")
    print("正确性验证")
    print(f"{'=' * 80}")
    print("  参考实现: single_prefill_with_kv_cache + custom_mask")
    print("  Mask 规则: mask[q,k] = ((q + offset) // B) >= (k // B)")

    # 计算参考结果 (每个 chunk 独立计算)
    ref_outputs = []
    for step_idx in range(num_chunks):
        q_offset = step_idx * baseline_chunk_size
        ref_out = compute_block_extend_reference(
            qs_chunks[step_idx],
            k_cumul_list[step_idx],
            v_cumul_list[step_idx],
            block_size=block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
        )
        ref_outputs.append(ref_out)

    # 计算 single_prefill_with_kv_cache 结果并验证
    v2_outputs = []
    for step_idx in range(num_chunks):
        q_offset = step_idx * baseline_chunk_size
        v2_out = single_prefill_with_kv_cache(
            qs_chunks[step_idx],
            k_cumul_list[step_idx],
            v_cumul_list[step_idx],
            block_extend=True,
            block_size=block_size,
            q_offset=q_offset,
            sm_scale=sm_scale,
            backend=backend,
        )
        v2_outputs.append(v2_out)

    # 验证 single_prefill_with_kv_cache
    v2_max_diff = max(
        (v2_outputs[i] - ref_outputs[i]).abs().max().item() for i in range(num_chunks)
    )
    tol = 1e-3
    v2_pass = v2_max_diff < tol
    print("\n  single_prefill_with_kv_cache(block_extend=True):")
    print(f"       max_diff = {v2_max_diff:.6f}, tolerance = {tol}")
    print(f"       {' PASS' if v2_pass else ' FAIL'}")

    # 计算 BatchPrefillWithRaggedKVCacheWrapper 结果并验证
    bbe_outputs = []
    workspace_verify = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_indptr_verify = torch.tensor(
        [0, baseline_chunk_size], dtype=torch.int32, device=device
    )
    for step_idx in range(num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
        q_offset_tensor = torch.tensor(
            [step_idx * baseline_chunk_size], dtype=torch.int32, device=device
        )

        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            workspace_verify,
            kv_layout="NHD",
            block_extend=True,
            block_size=block_size,
            backend=backend,
        )
        wrapper.plan(
            qo_indptr=qo_indptr_verify,
            kv_indptr=kv_indptr,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_data_type=dtype,
            sm_scale=sm_scale,
            causal=False,
            q_offsets=q_offset_tensor,
        )
        bbe_out = wrapper.run(
            qs_chunks[step_idx], k_cumul_list[step_idx], v_cumul_list[step_idx]
        )
        bbe_outputs.append(bbe_out)

    # 验证 BatchPrefillWithRaggedKVCacheWrapper
    bbe_max_diff = max(
        (bbe_outputs[i] - ref_outputs[i]).abs().max().item() for i in range(num_chunks)
    )
    bbe_pass = bbe_max_diff < tol
    print("\n  BatchPrefillWithRaggedKVCacheWrapper(block_extend=True):")
    print(f"        max_diff = {bbe_max_diff:.6f}, tolerance = {tol}")
    print(f"        {' PASS' if bbe_pass else ' FAIL'}")

    if not (v2_pass and bbe_pass):
        print("\n   正确性验证未通过，但继续执行性能测试")
    else:
        print("\n   所有方案正确性验证通过")

    del workspace_verify, bbe_outputs, v2_outputs, ref_outputs
    torch.cuda.empty_cache()

    # Baseline 2: Custom Mask (原生 single_prefill + custom_mask) + CUDA Graph
    print(f"\n{'=' * 80}")
    print("[Baseline 2] Custom Mask (single_prefill + custom_mask)")
    print(f"{'=' * 80}")
    print(f"  num_steps: {num_chunks}")
    print("  每个 step: single_prefill_with_kv_cache + custom_mask")
    print("  每个 step 内 kernel: 1 次 (但需要构造 mask tensor)")

    custom_mask_buffers = []
    for step_idx in range(num_chunks):
        kv_len = (step_idx + 1) * baseline_chunk_size
        q_offset = step_idx * baseline_chunk_size
        # mask[q, k] = ((q + offset) // B) >= (k // B)
        q_pos = torch.arange(baseline_chunk_size, device=device) + q_offset
        k_pos = torch.arange(kv_len, device=device)
        q_block = q_pos.unsqueeze(1) // block_size
        k_block = k_pos.unsqueeze(0) // block_size
        mask_2d = (q_block >= k_block).to(torch.uint8)
        custom_mask_buffers.append(mask_2d)

    cm_output = torch.empty(
        baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device
    )

    # 注意: custom_mask 只在 FA2 后端支持，FA3 不支持
    def run_custom_mask_pipeline():
        for step_idx in range(num_chunks):
            cm_output.copy_(
                single_prefill_with_kv_cache(
                    qs_chunks[step_idx],
                    k_cumul_list[step_idx],
                    v_cumul_list[step_idx],
                    custom_mask=custom_mask_buffers[step_idx],  # noqa: F821
                    sm_scale=sm_scale,
                    backend=backend,
                )
            )

    if verbose:
        print("  Step 流程预览:")
        for step_id in range(num_chunks):
            kv_len = (step_id + 1) * baseline_chunk_size
            print(
                f"    Step {step_id}: Q[{step_id * baseline_chunk_size}:{(step_id + 1) * baseline_chunk_size}] attend to KV[0:{kv_len}]"
            )

    # Warmup
    print("  Warmup...")
    for _ in range(warmup_iters):
        run_custom_mask_pipeline()
    torch.cuda.synchronize()

    # CUDA Graph capture
    print("  Capturing CUDA Graph...")
    cm_stream = torch.cuda.Stream()
    with torch.cuda.stream(cm_stream):
        run_custom_mask_pipeline()
    cm_stream.synchronize()

    cm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cm_graph, stream=cm_stream):
        run_custom_mask_pipeline()

    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cm_graph.replay()
    torch.cuda.synchronize()

    # Benchmark
    print("  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cm_graph.replay()
    torch.cuda.synchronize()
    cm_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

    results["custom_mask_baseline"] = {
        "time_cuda_graph_ms": cm_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": num_chunks,
        "method": "Custom Mask (Baseline 2)",
    }
    print(
        f"  => cuda_graph: {cm_cuda_graph_time:.3f} ms ({cm_cuda_graph_time / num_chunks:.3f} ms/step × {num_chunks} steps)"
    )

    del cm_graph, custom_mask_buffers
    torch.cuda.empty_cache()

    # Custom Mask 不同 chunk_size 测试 (不要求 chunk_size = block_size)
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(
                f"\n[跳过] chunk_size={test_chunk_size} 无法整除 tokens_per_request={tokens_per_request}"
            )
            continue

        # 跳过已经测试过的 baseline chunk_size
        if test_chunk_size == baseline_chunk_size:
            continue

        num_steps_cm = tokens_per_request // test_chunk_size

        print(f"\n{'-' * 60}")
        print(f"[Custom Mask] chunk_size = {test_chunk_size}")
        print(f"{'-' * 60}")
        print(
            f"  num_steps: {num_steps_cm} (比 Baseline 少 {num_chunks - num_steps_cm} 步)"
        )
        print("  每个 step 内 kernel: 1 次")

        # 分 chunk
        qs_cm = split_chunks(q_full, test_chunk_size)

        # 累积 KV buffer
        k_cumul_cm = [
            k_full[: (i + 1) * test_chunk_size].clone() for i in range(num_steps_cm)
        ]
        v_cumul_cm = [
            v_full[: (i + 1) * test_chunk_size].clone() for i in range(num_steps_cm)
        ]

        # 预分配 custom_mask buffers (每个 step 的 mask 不同)
        cm_mask_buffers = []
        for step_idx in range(num_steps_cm):
            kv_len = (step_idx + 1) * test_chunk_size
            q_offset = step_idx * test_chunk_size
            # mask[q, k] = ((q + offset) // B) >= (k // B)
            q_pos = torch.arange(test_chunk_size, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // block_size
            k_block = k_pos.unsqueeze(0) // block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)
            cm_mask_buffers.append(mask_2d)

        # 预分配 output buffer
        cm_output_var = torch.empty(
            test_chunk_size, num_heads, head_dim, dtype=dtype, device=device
        )

        def run_cm_pipeline():
            for step_idx in range(num_steps_cm):
                cm_output_var.copy_(
                    single_prefill_with_kv_cache(
                        qs_cm[step_idx],
                        k_cumul_cm[step_idx],
                        v_cumul_cm[step_idx],
                        custom_mask=cm_mask_buffers[step_idx],  # noqa: F821
                        sm_scale=sm_scale,
                        backend=backend,
                    )
                )

        # 显示 step 流程
        if verbose:
            print("  Step 流程预览:")
            for step_id in range(num_steps_cm):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(
                    f"    Step {step_id}: Q[{q_offset}:{q_offset + test_chunk_size}] attend to KV[0:{kv_len}]"
                )

        # Warmup
        print("  Warmup...")
        for _ in range(warmup_iters):
            run_cm_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        print("  Capturing CUDA Graph...")
        cm_stream_var = torch.cuda.Stream()
        with torch.cuda.stream(cm_stream_var):
            run_cm_pipeline()
        cm_stream_var.synchronize()

        cm_graph_var = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cm_graph_var, stream=cm_stream_var):
            run_cm_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()

        # Benchmark
        print("  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            cm_graph_var.replay()
        torch.cuda.synchronize()
        cm_var_time = (time.perf_counter() - start) / bench_iters * 1000

        results[f"cm_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": cm_var_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_steps_cm,
            "method": "Custom Mask",
        }
        print(
            f"  => cuda_graph: {cm_var_time:.3f} ms ({cm_var_time / num_steps_cm:.3f} ms/step × {num_steps_cm} steps)"
        )

        del cm_graph_var, cm_mask_buffers
        torch.cuda.empty_cache()

    # Baseline 1: SGLang 3阶段 DLLM Cascade (BatchPrefill 接口, batch_size=1)
    print(f"\n{'=' * 80}")
    print("[Baseline 1] SGLang 3阶段 Cascade (BatchPrefill, batch_size=1)")
    print(f"{'=' * 80}")
    print(f"  num_steps: {num_chunks}")
    print("  每个 step: BatchPrefill(当前 chunk) + BatchPrefill(prefix) + merge_state")
    print("  每个 step 内 kernel: 2-3 次")

    # batch_size=1 的 indptr
    qo_indptr_chunk = torch.tensor(
        [0, baseline_chunk_size], dtype=torch.int32, device=device
    )

    # workspace: 共享一个 buffer，避免每个 wrapper 独立分配导致 OOM
    ws_curr = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    ws_prefix = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    # 为每个 step 创建独立的 wrapper 并完成 plan（共享 workspace）
    print("  创建 wrappers 并完成 plan...")
    cascade_wrappers_current = []
    cascade_wrappers_prefix = []
    kv_indptr_prefix_list = [None]  # step 0 没有 prefix

    for step_idx in range(num_chunks):
        # 当前 chunk 的 wrapper
        wrapper_current = BatchPrefillWithRaggedKVCacheWrapper(
            ws_curr,
            kv_layout="NHD",
            backend=backend,
        )
        wrapper_current.plan(
            qo_indptr=qo_indptr_chunk,
            kv_indptr=qo_indptr_chunk,  # 当前 chunk KV 长度 = Q 长度
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            causal=False,
            sm_scale=sm_scale,
        )
        cascade_wrappers_current.append(wrapper_current)

        # prefix 的 wrapper (step 0 没有 prefix)
        if step_idx == 0:
            cascade_wrappers_prefix.append(None)
        else:
            prefix_len = step_idx * baseline_chunk_size
            kv_indptr_prefix = torch.tensor(
                [0, prefix_len], dtype=torch.int32, device=device
            )
            kv_indptr_prefix_list.append(kv_indptr_prefix)

            wrapper_prefix = BatchPrefillWithRaggedKVCacheWrapper(
                ws_prefix,
                kv_layout="NHD",
                backend=backend,
            )
            wrapper_prefix.plan(
                qo_indptr=qo_indptr_chunk,
                kv_indptr=kv_indptr_prefix,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=False,
                sm_scale=sm_scale,
            )
            cascade_wrappers_prefix.append(wrapper_prefix)

    # 预分配 buffer
    cascade_output = torch.empty(
        baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device
    )
    o1_buffer = torch.empty(
        baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device
    )
    s1_buffer = torch.empty(
        baseline_chunk_size, num_heads, dtype=torch.float32, device=device
    )
    o2_buffer = torch.empty(
        baseline_chunk_size, num_heads, head_dim, dtype=dtype, device=device
    )
    s2_buffer = torch.empty(
        baseline_chunk_size, num_heads, dtype=torch.float32, device=device
    )

    # 预分配 prefix KV buffer
    k_prefix_buffers = [None]
    v_prefix_buffers = [None]
    for step_idx in range(1, num_chunks):
        prefix_len = step_idx * baseline_chunk_size
        k_prefix_buffers.append(k_full[:prefix_len].clone())
        v_prefix_buffers.append(v_full[:prefix_len].clone())

    def run_cascade_pipeline():
        for step_idx in range(num_chunks):
            q_chunk = qs_chunks[step_idx]
            k_chunk = ks_chunks[step_idx]
            v_chunk = vs_chunks[step_idx]

            if step_idx == 0:
                cascade_output.copy_(
                    cascade_wrappers_current[step_idx].run(q_chunk, k_chunk, v_chunk)  # noqa: F821
                )
            else:
                o1, s1 = cascade_wrappers_current[step_idx].run_return_lse(  # noqa: F821
                    q_chunk, k_chunk, v_chunk
                )
                o1_buffer.copy_(o1)
                s1_buffer.copy_(s1)

                o2, s2 = cascade_wrappers_prefix[step_idx].run_return_lse(  # noqa: F821
                    q_chunk, k_prefix_buffers[step_idx], v_prefix_buffers[step_idx]
                )
                o2_buffer.copy_(o2)
                s2_buffer.copy_(s2)

                o, _ = merge_state(o1_buffer, s1_buffer, o2_buffer, s2_buffer)
                cascade_output.copy_(o)

    if verbose:
        print("  Step 流程预览:")
        for step_id in range(num_chunks):
            prefix_len = step_id * baseline_chunk_size
            if step_id == 0:
                print(
                    f"    Step {step_id}: current_chunk[{baseline_chunk_size}] only (no prefix)"
                )
            else:
                print(
                    f"    Step {step_id}: current_chunk[{baseline_chunk_size}] + prefix[{prefix_len}] + merge"
                )

    # Warmup
    print("  Warmup...")
    for _ in range(warmup_iters):
        run_cascade_pipeline()
    torch.cuda.synchronize()

    # CUDA Graph capture
    print("  Capturing CUDA Graph...")
    cascade_stream = torch.cuda.Stream()
    with torch.cuda.stream(cascade_stream):
        run_cascade_pipeline()
    cascade_stream.synchronize()

    cascade_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cascade_graph, stream=cascade_stream):
        run_cascade_pipeline()

    # Warmup with cuda_graph
    for _ in range(warmup_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()

    # Benchmark
    print("  Benchmark (with cuda_graph)...")
    start = time.perf_counter()
    for _ in range(bench_iters):
        cascade_graph.replay()
    torch.cuda.synchronize()
    cascade_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

    results["cascade_baseline"] = {
        "time_cuda_graph_ms": cascade_cuda_graph_time,
        "chunk_size": baseline_chunk_size,
        "num_steps": num_chunks,
        "method": "SGLang Cascade (Baseline 1)",
    }
    print(
        f"  => cuda_graph: {cascade_cuda_graph_time:.3f} ms ({cascade_cuda_graph_time / num_chunks:.3f} ms/step × {num_chunks} steps)"
    )

    del cascade_wrappers_current, cascade_wrappers_prefix, cascade_graph
    torch.cuda.empty_cache()

    # 方案 1: single_prefill_with_kv_cache + CUDA Graph
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            print(
                f"\n[跳过] chunk_size={test_chunk_size} 无法整除 tokens_per_request={tokens_per_request}"
            )
            continue

        num_steps = tokens_per_request // test_chunk_size

        print(f"\n{'-' * 60}")
        print(
            f"[single_prefill_with_kv_cache] chunk_size={test_chunk_size} (block_extend=True)"
        )
        print(f"{'-' * 60}")
        print(f"  num_steps: {num_steps} (比 Baseline 少 {num_chunks - num_steps} 步)")
        print("  每个 step 内 kernel: 1 次")

        # 分 chunk
        qs_v2 = split_chunks(q_full, test_chunk_size)

        # 预分配 output buffer
        v2_output = torch.empty(
            test_chunk_size, num_heads, head_dim, dtype=dtype, device=device
        )

        # 累积 KV buffer
        k_cumul_v2 = [
            k_full[: (i + 1) * test_chunk_size].clone() for i in range(num_steps)
        ]
        v_cumul_v2 = [
            v_full[: (i + 1) * test_chunk_size].clone() for i in range(num_steps)
        ]

        def run_v2_pipeline():
            for step_idx in range(num_steps):
                v2_output.copy_(
                    single_prefill_with_kv_cache(
                        qs_v2[step_idx],
                        k_cumul_v2[step_idx],
                        v_cumul_v2[step_idx],
                        block_extend=True,
                        block_size=block_size,
                        q_offset=step_idx * test_chunk_size,
                        sm_scale=sm_scale,
                        backend=backend,
                    )
                )

        # 显示 step 流程
        if verbose:
            print("  Step 流程预览:")
            for step_id in range(num_steps):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(
                    f"    Step {step_id}: Q[{q_offset}:{q_offset + test_chunk_size}] attend to KV[0:{kv_len}] (q_offset={q_offset})"
                )

        # Warmup
        print("  Warmup...")
        for _ in range(warmup_iters):
            run_v2_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        print("  Capturing CUDA Graph...")
        v2_stream = torch.cuda.Stream()
        with torch.cuda.stream(v2_stream):
            run_v2_pipeline()
        v2_stream.synchronize()

        v2_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(v2_graph, stream=v2_stream):
            run_v2_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            v2_graph.replay()
        torch.cuda.synchronize()

        # Benchmark
        print("  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            v2_graph.replay()
        torch.cuda.synchronize()
        v2_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

        results[f"single_prefill_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": v2_cuda_graph_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_steps,
            "method": "single_prefill_with_kv_cache (block_extend)",
        }
        print(
            f"  => cuda_graph: {v2_cuda_graph_time:.3f} ms ({v2_cuda_graph_time / num_steps:.3f} ms/step × {num_steps} steps)"
        )

        del v2_graph
        torch.cuda.empty_cache()

    # 方案 2: BatchPrefillWithRaggedKVCacheWrapper + CUDA Graph
    for test_chunk_size in chunk_sizes:
        if tokens_per_request % test_chunk_size != 0:
            continue

        num_steps = tokens_per_request // test_chunk_size

        print(f"\n{'-' * 60}")
        print(
            f"[BatchPrefillWithRaggedKVCacheWrapper] chunk_size={test_chunk_size} (block_extend=True)"
        )
        print(f"{'-' * 60}")
        print(f"  num_steps: {num_steps} (比 Baseline 少 {num_chunks - num_steps} 步)")
        print("  每个 step 内 kernel: 1 次")

        # 分 chunk
        qs_bbe = split_chunks(q_full, test_chunk_size)

        # batch_size=1 的 indptr
        qo_indptr_bbe = torch.tensor(
            [0, test_chunk_size], dtype=torch.int32, device=device
        )

        # 累积 KV buffer
        k_cumul_bbe = [
            k_full[: (i + 1) * test_chunk_size].clone() for i in range(num_steps)
        ]
        v_cumul_bbe = [
            v_full[: (i + 1) * test_chunk_size].clone() for i in range(num_steps)
        ]

        # 为每个 step 创建独立的 wrapper 并完成 plan（共享 workspace）
        print("  创建 wrappers 并完成 plan...")

        ws_bbe = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        bbe_wrappers = []
        for step_idx in range(num_steps):
            kv_len = (step_idx + 1) * test_chunk_size
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offset = torch.tensor(
                [step_idx * test_chunk_size], dtype=torch.int32, device=device
            )

            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                ws_bbe,
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend=backend,
            )
            wrapper.plan(
                qo_indptr=qo_indptr_bbe,
                kv_indptr=kv_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=q_offset,
            )
            bbe_wrappers.append(wrapper)

        # 预分配 output buffer
        bbe_output = torch.empty(
            test_chunk_size, num_heads, head_dim, dtype=dtype, device=device
        )

        def run_bbe_pipeline():
            for step_idx in range(num_steps):
                bbe_output.copy_(
                    bbe_wrappers[step_idx].run(  # noqa: F821
                        qs_bbe[step_idx], k_cumul_bbe[step_idx], v_cumul_bbe[step_idx]
                    )
                )

        # 显示 step 流程
        if verbose:
            print("  Step 流程预览:")
            for step_id in range(num_steps):
                kv_len = (step_id + 1) * test_chunk_size
                q_offset = step_id * test_chunk_size
                print(
                    f"    Step {step_id}: Q[{q_offset}:{q_offset + test_chunk_size}] attend to KV[0:{kv_len}] (q_offset={q_offset})"
                )

        # Warmup
        print("  Warmup...")
        for _ in range(warmup_iters):
            run_bbe_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        print("  Capturing CUDA Graph...")
        bbe_stream = torch.cuda.Stream()
        with torch.cuda.stream(bbe_stream):
            run_bbe_pipeline()
        bbe_stream.synchronize()

        bbe_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bbe_graph, stream=bbe_stream):
            run_bbe_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()

        # Benchmark
        print("  Benchmark (with cuda_graph)...")
        start = time.perf_counter()
        for _ in range(bench_iters):
            bbe_graph.replay()
        torch.cuda.synchronize()
        bbe_cuda_graph_time = (time.perf_counter() - start) / bench_iters * 1000

        results[f"ragged_prefill_chunk{test_chunk_size}"] = {
            "time_cuda_graph_ms": bbe_cuda_graph_time,
            "chunk_size": test_chunk_size,
            "num_steps": num_steps,
            "method": "BatchPrefillWithRaggedKVCacheWrapper (block_extend)",
        }
        print(
            f"  => cuda_graph: {bbe_cuda_graph_time:.3f} ms ({bbe_cuda_graph_time / num_steps:.3f} ms/step × {num_steps} steps)"
        )

        del bbe_wrappers, bbe_graph
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print("结果汇总 (单请求分步执行 + CUDA Graph)")
    print(f"{'=' * 80}")

    cm_baseline_r = results.get("custom_mask_baseline", {})
    cascade_baseline_r = results.get("cascade_baseline", {})
    cascade_skipped = cascade_baseline_r.get("skipped", False)

    cm_baseline_time = cm_baseline_r.get("time_cuda_graph_ms", float("nan"))
    cascade_baseline_time = cascade_baseline_r.get("time_cuda_graph_ms", float("nan"))

    # 表头说明
    print("\n说明:")
    print("  - Baseline 1: SGLang Cascade (BatchPrefill + merge_state)")
    print("  - Baseline 2: Custom Mask (single_prefill + custom_mask)")
    print("  - vs Base1: 相对于 SGLang Cascade 的加速比")
    print("  - vs Base2: 相对于 Custom Mask 的加速比")

    print(
        f"\n{'方案':<45} | {'chunk':>6} | {'steps':>6} | {'cuda_graph(ms)':>10} | {'ms/step':>10} | {'vs Base1':>10} | {'vs Base2':>10}"
    )
    print(
        f"{'-' * 45}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}"
    )

    # Baseline 1: SGLang Cascade
    if cascade_skipped:
        print(
            f"{'[Baseline 1] SGLang Cascade':<45} | {cascade_baseline_r['chunk_size']:>6} | {cascade_baseline_r['num_steps']:>6} | {'SKIPPED':>10} | {'-':>10} | {'-':>10} | {'-':>10}"
        )
    else:
        r = cascade_baseline_r
        print(
            f"{'[Baseline 1] SGLang Cascade':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {'1.00x':>10} | {cm_baseline_time / cascade_baseline_time:>9.2f}x"
        )

    # Baseline 2: Custom Mask
    r = cm_baseline_r
    if cascade_skipped:
        vs_base1_str = "-"
    else:
        vs_base1_str = f"{cascade_baseline_time / cm_baseline_time:>9.2f}x"
    print(
        f"{'[Baseline 2] Custom Mask':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {vs_base1_str:>10} | {'1.00x':>10}"
    )

    # Custom Mask 不同 chunk_size 结果 (按 chunk_size 递增排序)
    cm_keys = sorted(
        [k for k in results.keys() if k.startswith("cm_chunk")],
        key=lambda k: results[k]["chunk_size"],
    )
    for key in cm_keys:
        r = results[key]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        if cascade_skipped:
            speedup_vs_cascade_str = "-"
        else:
            speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
            speedup_vs_cascade_str = f"{speedup_vs_cascade:>9.2f}x"
        print(
            f"{'Custom Mask':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {speedup_vs_cascade_str:>10} | {speedup_vs_cm:>9.2f}x"
        )

    # single_prefill_with_kv_cache 结果 (按 chunk_size 递增排序)
    single_keys = sorted(
        [k for k in results.keys() if k.startswith("single_prefill_chunk")],
        key=lambda k: results[k]["chunk_size"],
    )
    for key in single_keys:
        r = results[key]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        if cascade_skipped:
            speedup_vs_cascade_str = "-"
        else:
            speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
            speedup_vs_cascade_str = f"{speedup_vs_cascade:>9.2f}x"
        print(
            f"{'single_prefill_with_kv_cache (block_extend)':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {speedup_vs_cascade_str:>10} | {speedup_vs_cm:>9.2f}x"
        )

    # BatchPrefillWithRaggedKVCacheWrapper 结果 (按 chunk_size 递增排序)
    ragged_keys = sorted(
        [k for k in results.keys() if k.startswith("ragged_prefill_chunk")],
        key=lambda k: results[k]["chunk_size"],
    )
    for key in ragged_keys:
        r = results[key]
        speedup_vs_cm = cm_baseline_time / r["time_cuda_graph_ms"]
        if cascade_skipped:
            speedup_vs_cascade_str = "-"
        else:
            speedup_vs_cascade = cascade_baseline_time / r["time_cuda_graph_ms"]
            speedup_vs_cascade_str = f"{speedup_vs_cascade:>9.2f}x"
        print(
            f"{'BatchPrefillWithRaggedKVCacheWrapper (block_extend)':<45} | {r['chunk_size']:>6} | {r['num_steps']:>6} | {r['time_cuda_graph_ms']:>10.3f} | {r['time_cuda_graph_ms'] / r['num_steps']:>10.3f} | {speedup_vs_cascade_str:>10} | {speedup_vs_cm:>9.2f}x"
        )

    return results


def test_fa2_fa3_block_extending_vs_causal(
    num_requests: int = 4,
    chunk_sizes: list = None,
    tokens_per_request: int = 2048,
    block_size: int = 32,
    num_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    warmup_iters: int = 10,
    bench_iters: int = 100,
    verbose: bool = False,
):
    """
    FA2 vs FA3 BlockExtend Mask vs FA3 Causal Mask 性能对比

    测试场景 (增量 Prefill):
      - 每个 request 的总 tokens 固定 (tokens_per_request)
      - 按 chunk_size 分多步执行增量 prefill
      - 每个 step 累积 KV，实现 Block Extend Mask
      - 使用 CUDA Graph 减少 kernel launch overhead

    对比方案:
      1. FA3 Causal Mask (baseline) - BatchPrefillWithRaggedKVCacheWrapper
      2. FA2 BlockExtend Mask - BatchPrefillWithRaggedKVCacheWrapper (backend="fa2")
      3. FA3 BlockExtend Mask - BatchPrefillWithRaggedKVCacheWrapper (backend="fa3")

    Mask 规则:
      - Causal: mask[q, k] = (q + q_offset) >= k
      - BlockExtend: mask[q, k] = ((q + q_offset) // B) >= (k // B)
    """
    if chunk_sizes is None:
        chunk_sizes = [32, 64, 128, 256]

    device = torch.device("cuda:0")
    dtype = torch.float16
    sm_scale = 1.0 / (head_dim**0.5)

    print(f"\n{'=' * 90}")
    print("FA2 vs FA3 BlockExtend Mask vs Causal Mask - 增量 Prefill 性能对比")
    print(f"{'=' * 90}")
    print("配置:")
    print(f"  num_requests        = {num_requests}")
    print(f"  tokens_per_request  = {tokens_per_request}")
    print(f"  block_size     = {block_size}")
    print(f"  chunk_sizes         = {chunk_sizes}")
    print(f"  num_heads           = {num_heads}")
    print(f"  num_kv_heads        = {num_kv_heads}")
    print(f"  head_dim            = {head_dim}")
    print("\n场景说明:")
    print(f"  - 固定总 tokens = {tokens_per_request}")
    print("  - 按 chunk_size 分多步执行增量 prefill")
    print("  - 使用 CUDA Graph 减少 overhead")

    # 数据准备，生成每个 request 的完整 Q, K, V
    all_qs = [
        torch.randn(tokens_per_request, num_heads, head_dim, dtype=dtype, device=device)
        for _ in range(num_requests)
    ]
    all_ks = [
        torch.randn(
            tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        for _ in range(num_requests)
    ]
    all_vs = [
        torch.randn(
            tokens_per_request, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        for _ in range(num_requests)
    ]

    def split_chunks(tensor, chunk_size):
        return [
            tensor[i * chunk_size : (i + 1) * chunk_size]
            for i in range(tensor.shape[0] // chunk_size)
        ]

    results = {}

    for chunk_size in chunk_sizes:
        if tokens_per_request % chunk_size != 0:
            print(
                f"\n[跳过] chunk_size={chunk_size} 无法整除 tokens_per_request={tokens_per_request}"
            )
            continue

        num_steps = tokens_per_request // chunk_size

        print(f"\n{'-' * 90}")
        print(f"chunk_size = {chunk_size}, num_steps = {num_steps}")
        print(f"{'-' * 90}")

        # 为每个 request 分 chunk
        qs_chunks = [split_chunks(q, chunk_size) for q in all_qs]

        # 预分配所有 step 的 buffer
        # Q buffers: 每个 step 的 Q concat
        q_buffers = []
        for step_idx in range(num_steps):
            q_list = [qs_chunks[req_idx][step_idx] for req_idx in range(num_requests)]
            q_buffers.append(torch.cat(q_list, dim=0))

        # KV buffers: 累积的 K, V
        k_buffers = []
        v_buffers = []
        for step_idx in range(num_steps):
            kv_len = (step_idx + 1) * chunk_size
            k_cumul_list = [all_ks[req_idx][:kv_len] for req_idx in range(num_requests)]
            v_cumul_list = [all_vs[req_idx][:kv_len] for req_idx in range(num_requests)]
            k_buffers.append(torch.cat(k_cumul_list, dim=0))
            v_buffers.append(torch.cat(v_cumul_list, dim=0))

        # indptrs
        qo_indptr = torch.tensor(
            [i * chunk_size for i in range(num_requests + 1)],
            dtype=torch.int32,
            device=device,
        )
        kv_indptr_list = []
        q_offsets_list = []
        for step_idx in range(num_steps):
            kv_len = (step_idx + 1) * chunk_size
            kv_indptr_list.append(
                torch.tensor(
                    [i * kv_len for i in range(num_requests + 1)],
                    dtype=torch.int32,
                    device=device,
                )
            )
            q_offset = step_idx * chunk_size
            q_offsets_list.append(
                torch.full((num_requests,), q_offset, dtype=torch.int32, device=device)
            )

        workspace_size = 256 * 1024 * 1024
        output_buffer = torch.empty(
            num_requests * chunk_size, num_heads, head_dim, dtype=dtype, device=device
        )

        # [0] 精度验证: FA2 vs FA3 BlockExtend
        print("  [精度验证] 对比 FA2 vs FA3 BlockExtend 输出...")

        # 创建临时 wrappers 进行精度验证
        fa2_verify_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(workspace_size, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            block_extend=True,
            block_size=block_size,
            backend="fa2",
        )
        fa3_verify_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            torch.empty(workspace_size, dtype=torch.uint8, device=device),
            kv_layout="NHD",
            block_extend=True,
            block_size=block_size,
            backend="fa3",
        )

        # 对每个 step 验证精度
        max_diff_all_steps = 0.0
        for step_idx in range(num_steps):
            fa2_verify_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=q_offsets_list[step_idx],
            )
            fa3_verify_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=q_offsets_list[step_idx],
            )

            fa2_out = fa2_verify_wrapper.run(
                q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
            )
            fa3_out = fa3_verify_wrapper.run(
                q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
            )

            step_max_diff = (fa2_out - fa3_out).abs().max().item()
            max_diff_all_steps = max(max_diff_all_steps, step_max_diff)

            if verbose:
                print(f"    step {step_idx}: max_diff = {step_max_diff:.6f}")

        # fp16 精度下约 0.001 差异属正常
        precision_ok = max_diff_all_steps < 0.01
        status = "PASS" if precision_ok else " FAIL"
        print(f"  [精度验证] FA2 vs FA3 max_diff = {max_diff_all_steps:.6f} {status}")

        if not precision_ok:
            print("  [警告] FA2 和 FA3 BlockExtend 输出差异过大，性能数据可能不可信！")

        del fa2_verify_wrapper, fa3_verify_wrapper
        torch.cuda.empty_cache()

        # [1] FA3 Causal Mask (baseline)
        print("  [FA3 Causal] 创建 wrappers...")
        causal_wrappers = []
        for step_idx in range(num_steps):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                backend="fa3",
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                causal=True,
                sm_scale=sm_scale,
            )
            causal_wrappers.append(wrapper)

        def run_causal_pipeline():
            for step_idx in range(num_steps):
                output_buffer.copy_(
                    causal_wrappers[step_idx].run(  # noqa: F821
                        q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
                    )
                )

        # Warmup
        for _ in range(warmup_iters):
            run_causal_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        causal_stream = torch.cuda.Stream()
        with torch.cuda.stream(causal_stream):
            run_causal_pipeline()
        causal_stream.synchronize()

        causal_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(causal_graph, stream=causal_stream):
            run_causal_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            causal_graph.replay()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            causal_graph.replay()
        torch.cuda.synchronize()
        fa3_causal_time = (time.perf_counter() - start) / bench_iters * 1000

        print(
            f"  [FA3 Causal]       {fa3_causal_time:.3f} ms ({fa3_causal_time / num_steps:.3f} ms/step × {num_steps} steps)"
        )

        del causal_wrappers, causal_graph
        torch.cuda.empty_cache()

        # [2] FA2 BlockExtend Mask
        print("  [FA2 BlockExtend] 创建 wrappers...")
        fa2_be_wrappers = []
        for step_idx in range(num_steps):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend="fa2",
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=q_offsets_list[step_idx],
            )
            fa2_be_wrappers.append(wrapper)

        def run_fa2_be_pipeline():
            for step_idx in range(num_steps):
                output_buffer.copy_(
                    fa2_be_wrappers[step_idx].run(  # noqa: F821
                        q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
                    )
                )

        # Warmup
        for _ in range(warmup_iters):
            run_fa2_be_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        fa2_stream = torch.cuda.Stream()
        with torch.cuda.stream(fa2_stream):
            run_fa2_be_pipeline()
        fa2_stream.synchronize()

        fa2_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fa2_graph, stream=fa2_stream):
            run_fa2_be_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            fa2_graph.replay()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            fa2_graph.replay()
        torch.cuda.synchronize()
        fa2_be_time = (time.perf_counter() - start) / bench_iters * 1000

        speedup_fa2_vs_causal = fa3_causal_time / fa2_be_time
        print(
            f"  [FA2 BlockExtend]  {fa2_be_time:.3f} ms ({fa2_be_time / num_steps:.3f} ms/step, {speedup_fa2_vs_causal:.2f}x vs FA3 Causal)"
        )

        del fa2_be_wrappers, fa2_graph
        torch.cuda.empty_cache()

        # ================================================================
        # [3] FA3 BlockExtend Mask
        # ================================================================
        print("  [FA3 BlockExtend] 创建 wrappers...")
        fa3_be_wrappers = []
        for step_idx in range(num_steps):
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(workspace_size, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend="fa3",
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr_list[step_idx],
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=q_offsets_list[step_idx],
            )
            fa3_be_wrappers.append(wrapper)

        def run_fa3_be_pipeline():
            for step_idx in range(num_steps):
                output_buffer.copy_(
                    fa3_be_wrappers[step_idx].run(  # noqa: F821
                        q_buffers[step_idx], k_buffers[step_idx], v_buffers[step_idx]
                    )
                )

        # Warmup
        for _ in range(warmup_iters):
            run_fa3_be_pipeline()
        torch.cuda.synchronize()

        # CUDA Graph capture
        fa3_stream = torch.cuda.Stream()
        with torch.cuda.stream(fa3_stream):
            run_fa3_be_pipeline()
        fa3_stream.synchronize()

        fa3_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(fa3_graph, stream=fa3_stream):
            run_fa3_be_pipeline()

        # Warmup with cuda_graph
        for _ in range(warmup_iters):
            fa3_graph.replay()
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            fa3_graph.replay()
        torch.cuda.synchronize()
        fa3_be_time = (time.perf_counter() - start) / bench_iters * 1000

        speedup_fa3_vs_causal = fa3_causal_time / fa3_be_time
        speedup_fa3_vs_fa2 = fa2_be_time / fa3_be_time
        print(
            f"  [FA3 BlockExtend]  {fa3_be_time:.3f} ms ({fa3_be_time / num_steps:.3f} ms/step, {speedup_fa3_vs_causal:.2f}x vs FA3 Causal, {speedup_fa3_vs_fa2:.2f}x vs FA2 BlockExtend)"
        )

        results[f"chunk{chunk_size}"] = {
            "chunk_size": chunk_size,
            "num_steps": num_steps,
            "fa2_fa3_max_diff": max_diff_all_steps,
            "precision_ok": precision_ok,
            "fa3_causal_ms": fa3_causal_time,
            "fa2_be_ms": fa2_be_time,
            "fa3_be_ms": fa3_be_time,
            "speedup_fa2_vs_causal": speedup_fa2_vs_causal,
            "speedup_fa3_vs_causal": speedup_fa3_vs_causal,
            "speedup_fa3_vs_fa2": speedup_fa3_vs_fa2,
        }

        del fa3_be_wrappers, fa3_graph
        torch.cuda.empty_cache()

    print(f"\n{'=' * 90}")
    print(
        f"结果汇总 (num_requests={num_requests}, tokens_per_request={tokens_per_request}, block_size={block_size})"
    )
    print(f"{'=' * 90}")

    print(
        f"\n{'chunk':>8} | {'steps':>5} | {'FA3 Causal':>12} | {'FA2 BlockExtend':>12} | {'FA3 BlockExtend':>12} | {'FA2/Causal':>10} | {'FA3/Causal':>10} | {'FA3/FA2':>10}"
    )
    print(
        f"{'-' * 8}-+-{'-' * 5}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}"
    )

    for key in sorted(results.keys(), key=lambda k: results[k]["chunk_size"]):
        r = results[key]
        print(
            f"{r['chunk_size']:>8} | {r['num_steps']:>5} | {r['fa3_causal_ms']:>10.3f}ms | {r['fa2_be_ms']:>10.3f}ms | {r['fa3_be_ms']:>10.3f}ms | {r['speedup_fa2_vs_causal']:>9.2f}x | {r['speedup_fa3_vs_causal']:>9.2f}x | {r['speedup_fa3_vs_fa2']:>9.2f}x"
        )

    print("\n说明:")
    print(
        f"  - 场景: 增量 Prefill，固定总 tokens = {tokens_per_request}，按 chunk_size 分多步执行"
    )
    print(
        "  - FA2/Causal: FA2 BlockExtend 相对于 FA3 Causal 的加速比 (>1 表示 FA2 BE 更快)"
    )
    print(
        "  - FA3/Causal: FA3 BlockExtend 相对于 FA3 Causal 的加速比 (>1 表示 FA3 BE 更快)"
    )
    print(
        "  - FA3/FA2: FA3 BlockExtend 相对于 FA2 BlockExtend 的加速比 (>1 表示 FA3 更快)"
    )
    print(
        "  - BlockExtend mask 比 Causal mask 计算量更少 (tile 级别跳过)，理论上应该更快"
    )

    return results


def test_dllm_precision_vs_custom_mask_fa2(
    verbose: bool = True,
    test_dtypes: list = None,
):
    """
    DLLM 组件精度测试: 与原生 Custom Mask FA2 实现进行对比

    参考实现 (Ground Truth):
      - 单请求: single_prefill_with_kv_cache + custom_mask (FA2)
      - 多请求: BatchPrefillWithRaggedKVCacheWrapper + custom_mask (FA2)

    被测对象 (第31-33行导入的三个 DLLM 组件):
      1. BatchPrefillWithRaggedKVCacheWrapper
      2. BatchPrefillWithPagedKVCacheWrapper
      3. single_prefill_with_kv_cache

    测试覆盖:
      - 数据类型: fp16, bf16
      - 不同的 block_size: [16, 32, 64, 128]
      - 不同的 qo_len: [32, 64, 128, 256]
      - 不同的 kv_len: [64, 128, 256, 512, 1024]
      - 不同的 q_offset: [0, 32, 64, 128]
      - 不同的 num_heads / num_kv_heads 组合
      - 不同的 head_dim: [64, 128]

    Mask 规则: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)
    """
    device = torch.device("cuda:0")
    backends = ["fa2", "fa3"]

    if test_dtypes is None:
        test_dtypes = [torch.float16, torch.bfloat16]

    # 不同数据类型的精度容差
    dtype_tolerances = {
        torch.float16: 1e-2,
        torch.bfloat16: 2e-2,  # FA3 bf16 tile accumulation order differs from FA2;
        # max_diff up to ~2 ULP (0.015625) is expected
    }

    dtype_names = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }

    # 测试参数组合
    test_configs = [
        # 基础测试: 不同 block_size
        {
            "block_size": 16,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 64,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 128,
            "qo_len": 128,
            "kv_len": 256,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 qo_len
        {
            "block_size": 32,
            "qo_len": 32,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 128,
            "kv_len": 256,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 256,
            "kv_len": 512,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 kv_len
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 64,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 256,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 512,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 1024,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 q_offset (模拟增量 prefill 的不同 step)
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 192,
            "q_offset": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 256,
            "q_offset": 128,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 320,
            "q_offset": 192,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 head 配置 (MHA vs GQA vs MQA)
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 32,
            "head_dim": 128,
        },  # MHA
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
        },  # GQA-8
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 1,
            "head_dim": 128,
        },  # MQA
        # 不同 head_dim
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 64,
        },
        {
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 16,
            "num_kv_heads": 4,
            "head_dim": 256,
        },
        # 边界条件测试
        {
            "block_size": 32,
            "qo_len": 1,
            "kv_len": 32,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },  # 单 query
        {
            "block_size": 32,
            "qo_len": 32,
            "kv_len": 32,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },  # qo_len == kv_len == block_size
        {
            "block_size": 64,
            "qo_len": 33,
            "kv_len": 97,
            "q_offset": 17,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },  # 非对齐边界
        # 长序列测试
        {
            "block_size": 32,
            "qo_len": 128,
            "kv_len": 2048,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "block_size": 64,
            "qo_len": 256,
            "kv_len": 4096,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
    ]

    # 多请求测试配置
    multi_req_configs = [
        # 基础多请求测试
        {
            "num_requests": 2,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "num_requests": 8,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 block_size
        {
            "num_requests": 4,
            "block_size": 16,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "num_requests": 4,
            "block_size": 64,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 q_offset (模拟增量 prefill step)
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 192,
            "q_offset": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 256,
            "q_offset": 128,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        # 不同 head 配置
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 32,
            "head_dim": 128,
        },  # MHA
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
        },  # GQA-8
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 64,
            "kv_len": 128,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 1,
            "head_dim": 128,
        },  # MQA
        # 长序列多请求测试
        {
            "num_requests": 4,
            "block_size": 32,
            "qo_len": 128,
            "kv_len": 1024,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        {
            "num_requests": 2,
            "block_size": 64,
            "qo_len": 256,
            "kv_len": 2048,
            "q_offset": 0,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
    ]

    for dtype in test_dtypes:
        dtype_name = dtype_names[dtype]
        tol = dtype_tolerances[dtype]

        print(f"\n{'=' * 120}")
        print(f"DLLM 组件精度测试: 与原生 Custom Mask FA2 对比 [{dtype_name.upper()}]")
        print(f"{'=' * 120}")
        print(
            "参考实现: single_prefill_with_kv_cache / BatchPrefillWithRaggedKVCacheWrapper + custom_mask (FA2)"
        )
        print("被测后端: FA2, FA3 (DLLM BlockWise 已支持两个后端)")
        print(f"数据类型: {dtype_name}")
        print("Mask 规则: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)")
        print(f"精度容差: {tol}")

        print(f"\n{'=' * 100}")
        print(f"[第一部分] 单请求精度测试 (FA2 & FA3 后端) [{dtype_name}]")
        print(f"{'=' * 100}")
        print("参考实现: single_prefill_with_kv_cache + custom_mask (FA2)")
        print("被测对象:")
        print("  1. BatchPrefillWithRaggedKVCacheWrapper (batch_size=1) - FA2 后端")
        print("  2. BatchPrefillWithRaggedKVCacheWrapper (batch_size=1) - FA3 后端")
        print("  3. single_prefill_with_kv_cache")

        single_req_results = []

        for cfg_idx, cfg in enumerate(test_configs):
            block_size = cfg["block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)

            # 生成测试数据
            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            # 参考实现: single_prefill_with_kv_cache + custom_mask (FA2)
            # 构造 custom_mask: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // block_size  # [qo_len, 1]
            k_block = k_pos.unsqueeze(0) // block_size  # [1, kv_len]
            mask_2d = (q_block >= k_block).to(torch.uint8)  # [qo_len, kv_len]

            ref_output = single_prefill_with_kv_cache(
                q,
                k,
                v,
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )

            result = {
                "config_idx": cfg_idx,
                "block_size": block_size,
                "qo_len": qo_len,
                "kv_len": kv_len,
                "q_offset": q_offset,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }

            # 被测 1 & 2: BatchPrefillWithRaggedKVCacheWrapper (FA2 和 FA3 后端)
            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=device)

            for backend in backends:
                workspace = torch.empty(
                    128 * 1024 * 1024, dtype=torch.uint8, device=device
                )
                wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    block_extend=True,
                    block_size=block_size,
                    backend=backend,
                )
                wrapper.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    q_data_type=dtype,
                    sm_scale=sm_scale,
                    causal=False,
                    q_offsets=q_offset_tensor,
                )
                bbe_output = wrapper.run(q, k, v)

                # 计算精度差异
                bbe_diff = (bbe_output - ref_output).abs().max().item()
                bbe_mean_diff = (bbe_output - ref_output).abs().mean().item()
                bbe_pass = bbe_diff < tol

                result[f"bbe_{backend}_max_diff"] = bbe_diff
                result[f"bbe_{backend}_mean_diff"] = bbe_mean_diff
                result[f"bbe_{backend}_pass"] = bbe_pass

                del workspace, wrapper

            # 被测 3: single_prefill_with_kv_cache (backend="fa2")
            v2_output = single_prefill_with_kv_cache(
                q,
                k,
                v,
                block_extend=True,
                block_size=block_size,
                q_offset=q_offset,
                sm_scale=sm_scale,
                backend="fa2",
            )

            # 计算 V2 精度差异
            v2_diff = (v2_output - ref_output).abs().max().item()
            v2_mean_diff = (v2_output - ref_output).abs().mean().item()
            v2_pass = v2_diff < tol

            result["v2_max_diff"] = v2_diff
            result["v2_mean_diff"] = v2_mean_diff
            result["v2_pass"] = v2_pass

            single_req_results.append(result)

            if verbose:
                fa2_status = "PASS" if result["bbe_fa2_pass"] else "FAIL"
                fa3_status = "PASS" if result["bbe_fa3_pass"] else "FAIL"
                v2_status = "PASS" if v2_pass else "FAIL"
                print(
                    f"\n  [Test {cfg_idx:02d}] B={block_size:3d}, qo={qo_len:4d}, kv={kv_len:4d}, "
                    f"q_off={q_offset:3d}, heads={num_heads}/{num_kv_heads}, dim={head_dim}"
                )
                print(
                    f"           BBE-FA2: max_diff={result['bbe_fa2_max_diff']:.6f}, mean_diff={result['bbe_fa2_mean_diff']:.6f} [{fa2_status}]"
                )
                print(
                    f"           BBE-FA3: max_diff={result['bbe_fa3_max_diff']:.6f}, mean_diff={result['bbe_fa3_mean_diff']:.6f} [{fa3_status}]"
                )
                print(
                    f"           V2:      max_diff={v2_diff:.6f}, mean_diff={v2_mean_diff:.6f} [{v2_status}]"
                )

            torch.cuda.empty_cache()

        # 单请求测试汇总
        print(f"\n{'-' * 100}")
        print(f"[单请求精度测试汇总] [{dtype_name}]")
        print(f"{'-' * 100}")

        total_tests = len(single_req_results)

        for backend in backends:
            pass_count = sum(1 for r in single_req_results if r[f"bbe_{backend}_pass"])
            max_diff_all = max(r[f"bbe_{backend}_max_diff"] for r in single_req_results)
            mean_diff_all = (
                sum(r[f"bbe_{backend}_mean_diff"] for r in single_req_results)
                / total_tests
            )
            print(
                f"  BatchPrefillWithRaggedKVCacheWrapper ({backend.upper()}): {pass_count}/{total_tests} PASS"
            )
            print(f"    max_diff (all tests): {max_diff_all:.6f}")
            print(f"    mean_diff (avg):      {mean_diff_all:.6f}")

        v2_pass_count = sum(1 for r in single_req_results if r["v2_pass"])
        v2_max_diff_all = max(r["v2_max_diff"] for r in single_req_results)
        v2_mean_diff_all = (
            sum(r["v2_mean_diff"] for r in single_req_results) / total_tests
        )
        print(f"  single_prefill_with_kv_cache: {v2_pass_count}/{total_tests} PASS")
        print(f"    max_diff (all tests): {v2_max_diff_all:.6f}")
        print(f"    mean_diff (avg):      {v2_mean_diff_all:.6f}")

        # 失败测试详情
        for backend in backends:
            failed = [r for r in single_req_results if not r[f"bbe_{backend}_pass"]]
            if failed:
                print(f"\n  [BBE-{backend.upper()} 失败测试详情]")
                for r in failed:
                    print(
                        f"    Test {r['config_idx']:02d}: B={r['block_size']}, qo={r['qo_len']}, kv={r['kv_len']}, "
                        f"q_off={r['q_offset']}, max_diff={r[f'bbe_{backend}_max_diff']:.6f}"
                    )

        failed_v2 = [r for r in single_req_results if not r["v2_pass"]]
        if failed_v2:
            print("\n  [V2 失败测试详情]")
            for r in failed_v2:
                print(
                    f"    Test {r['config_idx']:02d}: B={r['block_size']}, qo={r['qo_len']}, kv={r['kv_len']}, "
                    f"q_off={r['q_offset']}, max_diff={r['v2_max_diff']:.6f}"
                )

        # ═══════════════════════════════════════════════════════════════════════════════
        # 第二部分: 多请求精度测试 (FA2 & FA3 后端)
        # ═══════════════════════════════════════════════════════════════════════════════
        print(f"\n{'=' * 100}")
        print(f"[第二部分] 多请求精度测试 (FA2 & FA3 后端) [{dtype_name}]")
        print(f"{'=' * 100}")
        print("参考实现: BatchPrefillWithRaggedKVCacheWrapper + custom_mask (FA2)")
        print("被测对象:")
        print("  1. BatchPrefillWithRaggedKVCacheWrapper - FA2 后端")
        print("  2. BatchPrefillWithRaggedKVCacheWrapper - FA3 后端")

        multi_req_results = []

        for cfg_idx, cfg in enumerate(multi_req_configs):
            num_requests = cfg["num_requests"]
            block_size = cfg["block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)

            q_list = [
                torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
                for _ in range(num_requests)
            ]
            k_list = [
                torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                for _ in range(num_requests)
            ]
            v_list = [
                torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
                for _ in range(num_requests)
            ]

            q_batch = torch.cat(q_list, dim=0)
            k_batch = torch.cat(k_list, dim=0)
            v_batch = torch.cat(v_list, dim=0)

            # 构造 mask
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // block_size
            k_block = k_pos.unsqueeze(0) // block_size
            mask_2d = q_block >= k_block
            mask_flat = mask_2d.flatten()
            batch_mask = mask_flat.repeat(num_requests)

            qo_indptr = torch.tensor(
                [i * qo_len for i in range(num_requests + 1)],
                dtype=torch.int32,
                device=device,
            )
            kv_indptr = torch.tensor(
                [i * kv_len for i in range(num_requests + 1)],
                dtype=torch.int32,
                device=device,
            )

            # 参考实现
            ref_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                kv_layout="NHD",
                backend="fa2",
            )
            ref_wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                custom_mask=batch_mask,
                causal=False,
                sm_scale=sm_scale,
            )
            ref_output = ref_wrapper.run(q_batch, k_batch, v_batch)

            q_offsets = torch.full(
                (num_requests,), q_offset, dtype=torch.int32, device=device
            )
            result = {
                "config_idx": cfg_idx,
                "num_requests": num_requests,
                "block_size": block_size,
                "qo_len": qo_len,
                "kv_len": kv_len,
                "q_offset": q_offset,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }

            for backend in backends:
                bbe_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD",
                    block_extend=True,
                    block_size=block_size,
                    backend=backend,
                )
                bbe_wrapper.plan(
                    qo_indptr=qo_indptr,
                    kv_indptr=kv_indptr,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    q_data_type=dtype,
                    sm_scale=sm_scale,
                    q_offsets=q_offsets,
                    causal=False,
                )
                bbe_output = bbe_wrapper.run(q_batch, k_batch, v_batch)

                bbe_diff = (bbe_output - ref_output).abs().max().item()
                bbe_mean_diff = (bbe_output - ref_output).abs().mean().item()
                bbe_pass = bbe_diff < tol

                result[f"bbe_{backend}_max_diff"] = bbe_diff
                result[f"bbe_{backend}_mean_diff"] = bbe_mean_diff
                result[f"bbe_{backend}_pass"] = bbe_pass
                del bbe_wrapper

            multi_req_results.append(result)

            if verbose:
                fa2_status = "PASS" if result["bbe_fa2_pass"] else "FAIL"
                fa3_status = "PASS" if result["bbe_fa3_pass"] else "FAIL"
                print(
                    f"\n  [Test {cfg_idx:02d}] reqs={num_requests}, B={block_size:3d}, qo={qo_len:4d}, kv={kv_len:4d}, "
                    f"q_off={q_offset:3d}, heads={num_heads}/{num_kv_heads}, dim={head_dim}"
                )
                print(
                    f"           BBE-FA2: max_diff={result['bbe_fa2_max_diff']:.6f} [{fa2_status}]"
                )
                print(
                    f"           BBE-FA3: max_diff={result['bbe_fa3_max_diff']:.6f} [{fa3_status}]"
                )

            del ref_wrapper
            torch.cuda.empty_cache()

        # 多请求测试汇总
        print(f"\n[多请求精度测试汇总] [{dtype_name}]")
        total_tests = len(multi_req_results)
        for backend in backends:
            pass_count = sum(1 for r in multi_req_results if r[f"bbe_{backend}_pass"])
            max_diff_all = max(r[f"bbe_{backend}_max_diff"] for r in multi_req_results)
            print(
                f"  BBE ({backend.upper()}): {pass_count}/{total_tests} PASS, max_diff={max_diff_all:.6f}"
            )

        for backend in backends:
            failed = [r for r in multi_req_results if not r[f"bbe_{backend}_pass"]]
            if failed:
                print(f"\n  [BBE-{backend.upper()} 失败详情]")
                for r in failed:
                    print(
                        f"    Test {r['config_idx']:02d}: reqs={r['num_requests']}, B={r['block_size']}, max_diff={r[f'bbe_{backend}_max_diff']:.6f}"
                    )

        # 第三部分: Paged KV Cache 测试
        print(f"\n[第三部分] Paged KV Cache 精度测试 [{dtype_name}]")

        page_size = 16
        paged_configs = [
            {
                "block_size": 32,
                "qo_len": 64,
                "kv_len": 128,
                "q_offset": 0,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
            {
                "block_size": 32,
                "qo_len": 64,
                "kv_len": 128,
                "q_offset": 32,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
            {
                "block_size": 64,
                "qo_len": 128,
                "kv_len": 512,
                "q_offset": 0,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
            {
                "block_size": 32,
                "qo_len": 64,
                "kv_len": 128,
                "q_offset": 0,
                "num_heads": 32,
                "num_kv_heads": 4,
                "head_dim": 128,
            },
            {
                "block_size": 32,
                "qo_len": 128,
                "kv_len": 1024,
                "q_offset": 0,
                "num_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
            },
        ]

        paged_results = []

        for cfg_idx, cfg in enumerate(paged_configs):
            block_size = cfg["block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)

            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            num_pages = (kv_len + page_size - 1) // page_size
            kv_data = torch.randn(
                num_pages,
                2,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=dtype,
                device=device,
            )

            paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
            paged_kv_indptr = torch.tensor(
                [0, num_pages], dtype=torch.int32, device=device
            )
            last_page_len = (
                kv_len - (num_pages - 1) * page_size
                if kv_len % page_size != 0
                else page_size
            )
            paged_kv_last_page_len = torch.tensor(
                [last_page_len], dtype=torch.int32, device=device
            )

            k_continuous = kv_data[:, 0, :, :, :].reshape(-1, num_kv_heads, head_dim)[
                :kv_len
            ]
            v_continuous = kv_data[:, 1, :, :, :].reshape(-1, num_kv_heads, head_dim)[
                :kv_len
            ]

            # 参考实现
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // block_size
            k_block = k_pos.unsqueeze(0) // block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)

            ref_output = single_prefill_with_kv_cache(
                q,
                k_continuous,
                v_continuous,
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )

            result = {
                "config_idx": cfg_idx,
                "block_size": block_size,
                "qo_len": qo_len,
                "kv_len": kv_len,
                "q_offset": q_offset,
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "page_size": page_size,
            }

            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)

            for backend in backends:
                paged_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD",
                    block_extend=True,
                    block_size=block_size,
                    backend=backend,
                )
                paged_wrapper.plan(
                    qo_indptr=qo_indptr,
                    paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices,
                    paged_kv_last_page_len=paged_kv_last_page_len,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    page_size=page_size,
                    q_data_type=dtype,
                    sm_scale=sm_scale,
                    q_offsets=q_offsets,
                    causal=False,
                )
                paged_output = paged_wrapper.run(q, kv_data)

                paged_diff = (paged_output - ref_output).abs().max().item()
                paged_pass = paged_diff < tol
                result[f"paged_{backend}_max_diff"] = paged_diff
                result[f"paged_{backend}_pass"] = paged_pass
                del paged_wrapper

            paged_results.append(result)

            if verbose:
                fa2_status = "PASS" if result["paged_fa2_pass"] else "FAIL"
                fa3_status = "PASS" if result["paged_fa3_pass"] else "FAIL"
                print(
                    f"  [Test {cfg_idx:02d}] B={block_size:3d}, qo={qo_len:4d}, kv={kv_len:4d}, "
                    f"q_off={q_offset:3d} - FA2:{fa2_status}, FA3:{fa3_status}"
                )

            torch.cuda.empty_cache()

        # Paged 测试汇总
        print(f"\n[Paged KV Cache 精度测试汇总] [{dtype_name}]")
        for backend in backends:
            pass_count = sum(1 for r in paged_results if r[f"paged_{backend}_pass"])
            max_diff_all = max(r[f"paged_{backend}_max_diff"] for r in paged_results)
            print(
                f"  Paged ({backend.upper()}): {pass_count}/{len(paged_results)} PASS, max_diff={max_diff_all:.6f}"
            )

        for backend in backends:
            failed = [r for r in paged_results if not r[f"paged_{backend}_pass"]]
            if failed:
                print(f"\n  [Paged-{backend.upper()} 失败详情]")
                for r in failed:
                    print(
                        f"    Test {r['config_idx']:02d}: B={r['block_size']}, max_diff={r[f'paged_{backend}_max_diff']:.6f}"
                    )

        # 总结
        print(f"\n{'=' * 100}")
        print(f"精度测试总结 [{dtype_name}]")
        print(f"{'=' * 100}")

        print("\n  单请求测试:")
        for backend in backends:
            pass_count = sum(1 for r in single_req_results if r[f"bbe_{backend}_pass"])
            print(
                f"    BBE ({backend.upper()}): {pass_count}/{len(single_req_results)} PASS"
            )
        v2_pass_count = sum(1 for r in single_req_results if r["v2_pass"])
        print(f"    V2: {v2_pass_count}/{len(single_req_results)} PASS")

        print("\n  多请求测试:")
        for backend in backends:
            pass_count = sum(1 for r in multi_req_results if r[f"bbe_{backend}_pass"])
            print(
                f"    BBE ({backend.upper()}): {pass_count}/{len(multi_req_results)} PASS"
            )

        print("\n  Paged 测试:")
        for backend in backends:
            pass_count = sum(1 for r in paged_results if r[f"paged_{backend}_pass"])
            print(
                f"    Paged ({backend.upper()}): {pass_count}/{len(paged_results)} PASS"
            )

        # 总体结果
        all_single_pass = all(
            r["bbe_fa2_pass"] and r["bbe_fa3_pass"] and r["v2_pass"]
            for r in single_req_results
        )
        all_multi_pass = all(
            r["bbe_fa2_pass"] and r["bbe_fa3_pass"] for r in multi_req_results
        )
        all_paged_pass = all(
            r["paged_fa2_pass"] and r["paged_fa3_pass"] for r in paged_results
        )

        overall_pass = all_single_pass and all_multi_pass and all_paged_pass
        overall_status = "ALL TESTS PASSED" if overall_pass else "SOME TESTS FAILED"
        print(f"\n  总体结果: {overall_status}")

        # FA2 vs FA3 对比
        fa2_single_max = max(r["bbe_fa2_max_diff"] for r in single_req_results)
        fa3_single_max = max(r["bbe_fa3_max_diff"] for r in single_req_results)
        fa2_multi_max = max(r["bbe_fa2_max_diff"] for r in multi_req_results)
        fa3_multi_max = max(r["bbe_fa3_max_diff"] for r in multi_req_results)
        fa2_paged_max = max(r["paged_fa2_max_diff"] for r in paged_results)
        fa3_paged_max = max(r["paged_fa3_max_diff"] for r in paged_results)

        print("\n  FA2 vs FA3 max_diff:")
        print(f"    单请求: FA2={fa2_single_max:.6f}, FA3={fa3_single_max:.6f}")
        print(f"    多请求: FA2={fa2_multi_max:.6f}, FA3={fa3_multi_max:.6f}")
        print(f"    Paged:  FA2={fa2_paged_max:.6f}, FA3={fa3_paged_max:.6f}")

    return {
        "overall_pass": overall_pass,
    }


def test_heterogeneous_prefix_batch(
    verbose: bool = True,
    backend: str = "fa2",
):
    """
    异构 prefix 测试: 不同请求有不同的 prefix 长度

    场景复现:
      - Req 0: 已经 prefill 过，有 prefix (kv_len=128, q_offset=64)
      - Req 1: 新请求，没有 prefix (kv_len=32, q_offset=0)
      - 两个请求拼在一起传给 batch block-extend attention 算子

    测试目的:
      1. 验证算子是否支持异构 kv_len 输入
      2. 检查是否有访存越界问题
      3. 验证精度是否正确

    参考实现: 每个请求独立使用 custom_mask 计算，然后拼接
    """
    from flashinfer.prefill import single_prefill_with_kv_cache

    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2

    print(f"\n{'=' * 100}")
    print("异构 Prefix 测试: 不同请求有不同的 prefix 长度")
    print(f"{'=' * 100}")
    print(f"测试后端: {backend}")
    print(f"精度容差: {tol}")

    # 测试配置: 异构 prefix 场景
    # 每个 config 是一个请求列表，每个请求有不同的 qo_len, kv_len, q_offset
    test_configs = [
        # 场景 1: 一个有 prefix，一个没有
        {
            "name": "Req0(有prefix) + Req1(无prefix)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {
                    "qo_len": 64,
                    "kv_len": 128,
                    "q_offset": 64,
                },  # Req0: step_2, 已有 64 tokens prefix
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},  # Req1: step_0, 无 prefix
            ],
        },
        # 场景 2: 三个请求，不同 step
        {
            "name": "Req0(step_3) + Req1(step_0) + Req2(step_1)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 32, "kv_len": 128, "q_offset": 96},  # Req0: step_3
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},  # Req1: step_0
                {"qo_len": 32, "kv_len": 64, "q_offset": 32},  # Req2: step_1
            ],
        },
        # 场景 3: 两个请求，kv_len 差异更大
        {
            "name": "Req0(kv=256) + Req1(kv=32)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 32, "kv_len": 256, "q_offset": 224},  # Req0: 很长的 prefix
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},  # Req1: 无 prefix
            ],
        },
        # 场景 4: 两个请求，qo_len 也不同
        {
            "name": "Req0(qo=64,kv=128) + Req1(qo=32,kv=32)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 128, "q_offset": 64},  # Req0
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},  # Req1
            ],
        },
        # 场景 5: 四个请求，混合场景
        {
            "name": "4个请求混合场景",
            "block_size": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 256, "q_offset": 192},  # Req0: step_3
                {"qo_len": 64, "kv_len": 64, "q_offset": 0},  # Req1: step_0
                {"qo_len": 64, "kv_len": 128, "q_offset": 64},  # Req2: step_1
                {"qo_len": 64, "kv_len": 192, "q_offset": 128},  # Req3: step_2
            ],
        },
        # 场景 6: 类似 SGLang batch 推理 - 长 prompt + 短 prompt
        {
            "name": "长短 prompt 混合(512 vs 32)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {
                    "qo_len": 512,
                    "kv_len": 512,
                    "q_offset": 0,
                },  # Req0: 长 prompt (如数学题)
                {
                    "qo_len": 32,
                    "kv_len": 32,
                    "q_offset": 0,
                },  # Req1: 短 prompt (如 "Say hello")
            ],
        },
        # 场景 7: 极端差异 - 超长 vs 超短
        {
            "name": "极端差异(1024 vs 16)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 1024, "kv_len": 1024, "q_offset": 0},  # Req0: 超长 prompt
                {"qo_len": 16, "kv_len": 16, "q_offset": 0},  # Req1: 超短 prompt
            ],
        },
        # 场景 8: 三个请求，长度递增
        {
            "name": "三个请求长度递增(64,256,512)",
            "block_size": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"qo_len": 64, "kv_len": 64, "q_offset": 0},  # Req0: 短
                {"qo_len": 256, "kv_len": 256, "q_offset": 0},  # Req1: 中
                {"qo_len": 512, "kv_len": 512, "q_offset": 0},  # Req2: 长
            ],
        },
        # 场景 9: 混合 prefill 阶段 + 不同 prompt 长度
        {
            "name": "混合prefill阶段+不同prompt长度",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {
                    "qo_len": 32,
                    "kv_len": 512,
                    "q_offset": 480,
                },  # Req0: 长 prompt, step_15
                {"qo_len": 32, "kv_len": 32, "q_offset": 0},  # Req1: 短 prompt, step_0
                {
                    "qo_len": 32,
                    "kv_len": 128,
                    "q_offset": 96,
                },  # Req2: 中 prompt, step_3
            ],
        },
    ]

    results = []
    all_pass = True

    for cfg_idx, cfg in enumerate(test_configs):
        block_size = cfg["block_size"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        requests = cfg["requests"]
        sm_scale = 1.0 / math.sqrt(head_dim)

        print(f"\n  [Test {cfg_idx:02d}] {cfg['name']}")
        print(
            f"           B={block_size}, heads={num_heads}/{num_kv_heads}, dim={head_dim}"
        )

        # 生成每个请求的数据
        qs = []
        ks = []
        vs = []
        for req in requests:
            qo_len = req["qo_len"]
            kv_len = req["kv_len"]
            qs.append(
                torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            )
            ks.append(
                torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            )
            vs.append(
                torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            )

        # 参考实现: 每个请求独立计算
        ref_outputs = []
        for req_idx, req in enumerate(requests):
            qo_len = req["qo_len"]
            kv_len = req["kv_len"]
            q_offset = req["q_offset"]

            # 构造 custom_mask
            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            q_block = q_pos.unsqueeze(1) // block_size
            k_block = k_pos.unsqueeze(0) // block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)

            ref_output = single_prefill_with_kv_cache(
                qs[req_idx],
                ks[req_idx],
                vs[req_idx],
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )
            ref_outputs.append(ref_output)

        # 拼接参考输出
        ref_output_cat = torch.cat(ref_outputs, dim=0)

        # 构造 batch 输入
        q_cat = torch.cat(qs, dim=0)
        k_cat = torch.cat(ks, dim=0)
        v_cat = torch.cat(vs, dim=0)

        # 构造 indptr
        qo_lens = [req["qo_len"] for req in requests]
        kv_lens = [req["kv_len"] for req in requests]
        q_offsets_list = [req["q_offset"] for req in requests]

        qo_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(qo_lens), dim=0).numpy()),
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(kv_lens), dim=0).numpy()),
            dtype=torch.int32,
            device=device,
        )
        q_offsets = torch.tensor(q_offsets_list, dtype=torch.int32, device=device)

        if verbose:
            print(f"           qo_indptr: {qo_indptr.tolist()}")
            print(f"           kv_indptr: {kv_indptr.tolist()}")
            print(f"           q_offsets: {q_offsets.tolist()}")

        # 被测对象: BatchPrefillWithRaggedKVCacheWrapper
        try:
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                workspace,
                kv_layout="NHD",
                block_extend=True,
                block_size=block_size,
                backend=backend,
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                num_qo_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_data_type=dtype,
                sm_scale=sm_scale,
                causal=False,
                q_offsets=q_offsets,
            )
            bbe_output = wrapper.run(q_cat, k_cat, v_cat)

            # 计算精度差异
            max_diff = (bbe_output - ref_output_cat).abs().max().item()
            mean_diff = (bbe_output - ref_output_cat).abs().mean().item()
            passed = max_diff < tol

            status = "PASS" if passed else "FAIL"
            print(
                f"           BBE-{backend.upper()}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]"
            )

            if not passed:
                all_pass = False
                # 详细打印每个请求的差异
                start_idx = 0
                for req_idx, req in enumerate(requests):
                    qo_len = req["qo_len"]
                    end_idx = start_idx + qo_len
                    req_diff = (
                        (bbe_output[start_idx:end_idx] - ref_outputs[req_idx])
                        .abs()
                        .max()
                        .item()
                    )
                    print(
                        f"             Req{req_idx} (qo={qo_len}, kv={req['kv_len']}, q_off={req['q_offset']}): max_diff={req_diff:.6f}"
                    )
                    start_idx = end_idx

            results.append(
                {
                    "config_idx": cfg_idx,
                    "name": cfg["name"],
                    "max_diff": max_diff,
                    "passed": passed,
                    "error": None,
                }
            )

            del workspace, wrapper

        except Exception as e:
            print(f"           BBE-{backend.upper()}: ERROR - {str(e)}")
            results.append(
                {
                    "config_idx": cfg_idx,
                    "name": cfg["name"],
                    "max_diff": float("inf"),
                    "passed": False,
                    "error": str(e),
                }
            )
            all_pass = False

        torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'=' * 100}")
    print("异构 Prefix 测试汇总")
    print(f"{'=' * 100}")

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"  通过: {passed_count}/{total_count}")

    if not all_pass:
        print("\n  失败的测试:")
        for r in results:
            if not r["passed"]:
                if r["error"]:
                    print(
                        f"    - [{r['config_idx']:02d}] {r['name']}: ERROR - {r['error']}"
                    )
                else:
                    print(
                        f"    - [{r['config_idx']:02d}] {r['name']}: max_diff={r['max_diff']:.6f}"
                    )

    return results


def test_cascade_current_chunk_batch(
    verbose: bool = True,
    backend: str = "fa2",
):
    """
    测试完整的三阶段 Cascade Attention（模拟 SGLang DLLM 流程）

    三阶段流程:
      Stage 1 (prefix): BatchPrefillWithRaggedKVCacheWrapper 算 prefix
        - K/V: [0, prefix_len)
        - kv_offset = 0
      Stage 2 (current chunk): BatchPrefillWithRaggedKVCacheWrapper 算当前 chunk
        - K/V: [prefix_len, prefix_len + chunk_len)（只有当前 chunk）
        - kv_offset = prefix_len
      Stage 3 (merge): merge_state(o1, s1, o2, s2)

    Mask 规则: mask[q, k] = (q_global // B) >= (k_global // B)

    关键点:
      - Stage 2 的 K/V 不是从 0 开始，需要 kv_offset
      - 使用 blockwise extend mask（不是 causal mask）
    """
    device = "cuda"
    dtype = torch.bfloat16
    tol = 0.01 if backend == "fa3" else 0.01

    print(f"\n{'=' * 100}")
    print("三阶段 Cascade Attention 测试 (prefix + current_chunk + merge)")
    print(f"{'=' * 100}")
    print(f"测试后端: {backend}")
    print(f"精度容差: {tol}")

    # 测试配置: 每个请求有 prefix_len 和 chunk_len
    test_configs = [
        # 场景 1: 两个请求，一个有 prefix，一个没有
        {
            "name": "Req0(有prefix) + Req1(无prefix)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {
                    "prefix_len": 64,
                    "chunk_len": 32,
                },  # Req0: step_2, prefix [0,64), chunk [64,96)
                {
                    "prefix_len": 0,
                    "chunk_len": 32,
                },  # Req1: step_0, 无 prefix, chunk [0,32)
            ],
        },
        # 场景 2: 三个请求，不同 step
        {
            "name": "Req0(step_3) + Req1(step_0) + Req2(step_1)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 96, "chunk_len": 32},  # Req0: step_3
                {"prefix_len": 0, "chunk_len": 32},  # Req1: step_0
                {"prefix_len": 32, "chunk_len": 32},  # Req2: step_1
            ],
        },
        # 场景 3: 大 prefix
        {
            "name": "Req0(大prefix=256) + Req1(无prefix)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 256, "chunk_len": 32},  # Req0: step_8
                {"prefix_len": 0, "chunk_len": 32},  # Req1: step_0
            ],
        },
        # 场景 4: chunk_len != block_size
        {
            "name": "Req0(chunk=64) + Req1(chunk=32)",
            "block_size": 32,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 64, "chunk_len": 64},  # Req0: 2 blocks chunk
                {"prefix_len": 0, "chunk_len": 32},  # Req1: 1 block chunk
            ],
        },
        # 场景 5: 四个请求混合
        {
            "name": "4个请求混合(step_0,1,2,3)",
            "block_size": 64,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "requests": [
                {"prefix_len": 0, "chunk_len": 64},  # Req0: step_0
                {"prefix_len": 64, "chunk_len": 64},  # Req1: step_1
                {"prefix_len": 128, "chunk_len": 64},  # Req2: step_2
                {"prefix_len": 192, "chunk_len": 64},  # Req3: step_3
            ],
        },
    ]

    results = []
    all_pass = True

    for cfg_idx, cfg in enumerate(test_configs):
        block_size = cfg["block_size"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        requests = cfg["requests"]
        sm_scale = 1.0 / math.sqrt(head_dim)

        print(f"\n  [Test {cfg_idx:02d}] {cfg['name']}")
        print(
            f"           B={block_size}, heads={num_heads}/{num_kv_heads}, dim={head_dim}"
        )

        # 生成每个请求的数据
        # 每个请求有: Q(chunk_len), K_prefix(prefix_len), V_prefix(prefix_len), K_chunk(chunk_len), V_chunk(chunk_len)
        qs = []  # Q: 当前 chunk 的 query
        k_prefixes = []  # K_prefix: prefix 部分的 K
        v_prefixes = []  # V_prefix: prefix 部分的 V
        k_chunks = []  # K_chunk: 当前 chunk 的 K
        v_chunks = []  # V_chunk: 当前 chunk 的 V

        for req in requests:
            prefix_len = req["prefix_len"]
            chunk_len = req["chunk_len"]
            qs.append(
                torch.randn(chunk_len, num_heads, head_dim, dtype=dtype, device=device)
            )
            if prefix_len > 0:
                k_prefixes.append(
                    torch.randn(
                        prefix_len, num_kv_heads, head_dim, dtype=dtype, device=device
                    )
                )
                v_prefixes.append(
                    torch.randn(
                        prefix_len, num_kv_heads, head_dim, dtype=dtype, device=device
                    )
                )
            else:
                k_prefixes.append(None)
                v_prefixes.append(None)
            k_chunks.append(
                torch.randn(
                    chunk_len, num_kv_heads, head_dim, dtype=dtype, device=device
                )
            )
            v_chunks.append(
                torch.randn(
                    chunk_len, num_kv_heads, head_dim, dtype=dtype, device=device
                )
            )

        # 参考实现: 每个请求独立计算（完整 KV + blockwise mask）
        ref_outputs = []
        for req_idx, req in enumerate(requests):
            prefix_len = req["prefix_len"]
            chunk_len = req["chunk_len"]
            total_kv_len = prefix_len + chunk_len

            # 拼接完整的 K/V
            if prefix_len > 0:
                k_full = torch.cat([k_prefixes[req_idx], k_chunks[req_idx]], dim=0)
                v_full = torch.cat([v_prefixes[req_idx], v_chunks[req_idx]], dim=0)
            else:
                k_full = k_chunks[req_idx]
                v_full = v_chunks[req_idx]

            # 构造 blockwise extend mask
            # Q: [prefix_len, prefix_len + chunk_len)
            # K: [0, prefix_len + chunk_len)
            q_offset = prefix_len
            q_pos = torch.arange(chunk_len, device=device) + q_offset
            k_pos = torch.arange(total_kv_len, device=device)  # K 从 0 开始
            q_block = q_pos.unsqueeze(1) // block_size
            k_block = k_pos.unsqueeze(0) // block_size
            mask_2d = (q_block >= k_block).to(torch.uint8)

            ref_output = single_prefill_with_kv_cache(
                qs[req_idx],
                k_full,
                v_full,
                custom_mask=mask_2d,
                sm_scale=sm_scale,
                backend="fa2",
            )
            ref_outputs.append(ref_output)

        ref_output_cat = torch.cat(ref_outputs, dim=0)

        if verbose:
            for req_idx, req in enumerate(requests):
                prefix_len = req["prefix_len"]
                chunk_len = req["chunk_len"]
                print(
                    f"           Req{req_idx}: prefix_len={prefix_len}, chunk_len={chunk_len}, q_offset={prefix_len}, kv_offset={prefix_len}"
                )

        # 被测对象: 三阶段 Cascade Attention
        try:
            cascade_outputs = []
            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

            for req_idx, req in enumerate(requests):
                prefix_len = req["prefix_len"]
                chunk_len = req["chunk_len"]
                q = qs[req_idx]

                # Stage 1: prefix (if exists)
                if prefix_len > 0:
                    # 构造 prefix 的 indptr
                    prefix_qo_indptr = torch.tensor(
                        [0, chunk_len], dtype=torch.int32, device=device
                    )
                    prefix_kv_indptr = torch.tensor(
                        [0, prefix_len], dtype=torch.int32, device=device
                    )
                    prefix_q_offsets = torch.tensor(
                        [prefix_len], dtype=torch.int32, device=device
                    )  # Q 的全局位置
                    prefix_kv_offsets = torch.tensor(
                        [0], dtype=torch.int32, device=device
                    )  # prefix K/V 从 0 开始

                    prefix_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                        workspace,
                        kv_layout="NHD",
                        block_extend=True,
                        block_size=block_size,
                        backend=backend,
                    )
                    prefix_wrapper.plan(
                        qo_indptr=prefix_qo_indptr,
                        kv_indptr=prefix_kv_indptr,
                        num_qo_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim_qk=head_dim,
                        q_data_type=dtype,
                        sm_scale=sm_scale,
                        causal=False,
                        q_offsets=prefix_q_offsets,
                        kv_offsets=prefix_kv_offsets,
                    )
                    o1, s1 = prefix_wrapper.run(
                        q, k_prefixes[req_idx], v_prefixes[req_idx], return_lse=True
                    )
                    del prefix_wrapper
                else:
                    o1 = None
                    s1 = None

                # Stage 2: current chunk
                chunk_qo_indptr = torch.tensor(
                    [0, chunk_len], dtype=torch.int32, device=device
                )
                chunk_kv_indptr = torch.tensor(
                    [0, chunk_len], dtype=torch.int32, device=device
                )
                chunk_q_offsets = torch.tensor(
                    [prefix_len], dtype=torch.int32, device=device
                )  # Q 的全局位置
                chunk_kv_offsets = torch.tensor(
                    [prefix_len], dtype=torch.int32, device=device
                )  # chunk K/V 的全局位置

                chunk_wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                    workspace,
                    kv_layout="NHD",
                    block_extend=True,
                    block_size=block_size,
                    backend=backend,
                )
                chunk_wrapper.plan(
                    qo_indptr=chunk_qo_indptr,
                    kv_indptr=chunk_kv_indptr,
                    num_qo_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim_qk=head_dim,
                    q_data_type=dtype,
                    sm_scale=sm_scale,
                    causal=False,
                    q_offsets=chunk_q_offsets,
                    kv_offsets=chunk_kv_offsets,
                )
                o2, s2 = chunk_wrapper.run(
                    q, k_chunks[req_idx], v_chunks[req_idx], return_lse=True
                )
                del chunk_wrapper

                # Stage 3: merge
                if o1 is not None:
                    o_merged, _ = merge_state(o1, s1, o2, s2)
                    cascade_outputs.append(o_merged)
                else:
                    cascade_outputs.append(o2)

            cascade_output_cat = torch.cat(cascade_outputs, dim=0)

            # 计算精度差异
            max_diff = (cascade_output_cat - ref_output_cat).abs().max().item()
            mean_diff = (cascade_output_cat - ref_output_cat).abs().mean().item()
            passed = max_diff < tol

            status = "PASS" if passed else "FAIL"
            print(
                f"           Cascade-{backend.upper()}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]"
            )

            if not passed:
                all_pass = False

            results.append(
                {
                    "config_idx": cfg_idx,
                    "name": cfg["name"],
                    "max_diff": max_diff,
                    "passed": passed,
                    "error": None,
                }
            )

            del workspace

        except Exception as e:
            import traceback

            print(f"           Cascade-{backend.upper()}: ERROR - {str(e)}")
            if verbose:
                traceback.print_exc()
            results.append(
                {
                    "config_idx": cfg_idx,
                    "name": cfg["name"],
                    "max_diff": float("inf"),
                    "passed": False,
                    "error": str(e),
                }
            )
            all_pass = False

        torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'=' * 100}")
    print("三阶段 Cascade Attention 测试汇总")
    print(f"{'=' * 100}")

    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    print(f"  通过: {passed_count}/{total_count}")

    if not all_pass:
        print("\n  失败的测试:")
        for r in results:
            if not r["passed"]:
                if r["error"]:
                    print(
                        f"    - [{r['config_idx']:02d}] {r['name']}: ERROR - {r['error']}"
                    )
                else:
                    print(
                        f"    - [{r['config_idx']:02d}] {r['name']}: max_diff={r['max_diff']:.6f}"
                    )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cascade vs Batch 对比测试")
    parser.add_argument(
        "--batch-prefill",
        action="store_true",
        help="多 Request Batch Prefill 对比 (chunk_size 可变)",
    )
    parser.add_argument(
        "--step-by-step", action="store_true", help="真实分步执行对比 (模拟流水线依赖)"
    )
    parser.add_argument(
        "--step-by-step-cuda_graph",
        action="store_true",
        help="真实分步执行对比 + CUDA Graph",
    )
    parser.add_argument(
        "--single-req-cuda_graph",
        action="store_true",
        help="单请求分步执行 + CUDA Graph (流水线依赖)",
    )
    parser.add_argument(
        "--fa2-fa3-be",
        action="store_true",
        help="FA2 vs FA3 BlockExtend vs Causal 性能对比",
    )
    parser.add_argument(
        "--heterogeneous-prefix",
        action="store_true",
        help="异构 prefix 测试: 不同请求有不同的 prefix 长度",
    )
    parser.add_argument(
        "--cascade-chunk",
        action="store_true",
        help="Cascade Current Chunk 测试: K/V 只有当前 block，需要 kv_offset",
    )
    parser.add_argument(
        "--precision-test",
        action="store_true",
        help="DLLM 组件精度测试 (与原生 Custom Mask FA2 对比)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "all"],
        help="精度测试的数据类型: fp16, bf16, 或 all (同时测试两者)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="显示详细的 Step 流程预览信息"
    )
    parser.add_argument("--num_chunks", type=int, default=8)
    parser.add_argument(
        "--chunk_len", type=int, default=32, help="chunk 长度 (prefill seq_len)"
    )
    parser.add_argument(
        "--dllm_block_size",
        type=int,
        default=None,
        help="DLLM block size (默认 = chunk_len)",
    )
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_kv_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument(
        "--total_tokens",
        type=int,
        default=256,
        help="总 token 数 (用于 fair/width 模式)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (用于 --multi 模式)"
    )
    parser.add_argument("--num_requests", type=int, default=4, help="request 数量")
    parser.add_argument(
        "--tokens_per_request", type=int, default=256, help="每个 request 的 tokens"
    )
    parser.add_argument(
        "--kv_len",
        type=int,
        default=2048,
        help="总 tokens 数 (用于 --fa2-fa3-be 模式，即 tokens_per_request)",
    )
    parser.add_argument(
        "--chunk_sizes",
        type=str,
        default="32,64,128,256,512",
        help="chunk sizes 列表，逗号分隔",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fa2",
        choices=["auto", "fa2", "fa3"],
        help="后端实现: auto/fa2/fa3",
    )
    args = parser.parse_args()

    dllm_bs = (
        args.dllm_block_size if args.dllm_block_size is not None else args.chunk_len
    )

    if args.fa2_fa3_be:
        # FA2 vs FA3 BlockExtend vs Causal 性能对比 (增量 Prefill 场景)
        chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
        test_fa2_fa3_block_extending_vs_causal(
            num_requests=args.num_requests,
            chunk_sizes=chunk_sizes,
            tokens_per_request=args.kv_len,  # 使用 kv_len 参数作为 tokens_per_request
            block_size=dllm_bs,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
        )
    elif args.step_by_step_cuda_graph:
        # 多请求真实分步执行对比 + CUDA Graph
        test_incremental_batchprefill_step_by_step_with_cuda_graph(
            num_requests=args.num_requests,
            tokens_per_request=args.tokens_per_request,
            block_size=dllm_bs,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
            backend=args.backend,
        )
    elif args.single_req_cuda_graph:
        # 单请求分步执行 + CUDA Graph (流水线依赖)
        test_incremental_singlereq_prefill_step_by_step_with_cuda_graph(
            tokens_per_request=args.tokens_per_request,
            block_size=dllm_bs,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            verbose=args.verbose,
            backend=args.backend,
        )
    elif args.precision_test:
        if args.dtype == "all":
            print("\n" + "=" * 120)
            print("进行 FP16 精度测试...")
            print("=" * 120)
            test_dllm_precision_vs_custom_mask_fa2(
                verbose=args.verbose,
                test_dtypes=[torch.float16],
            )
            print("\n" + "=" * 120)
            print("进行 BF16 精度测试...")
            print("=" * 120)
            test_dllm_precision_vs_custom_mask_fa2(
                verbose=args.verbose,
                test_dtypes=[torch.bfloat16],
            )
        else:
            if args.dtype == "fp16":
                test_dtypes = [torch.float16]
            else:  # bf16
                test_dtypes = [torch.bfloat16]
            test_dllm_precision_vs_custom_mask_fa2(
                verbose=args.verbose,
                test_dtypes=test_dtypes,
            )
    elif args.heterogeneous_prefix:
        # 异构 prefix 测试：不同请求有不同的 prefix 长度
        for be in ["fa2", "fa3"]:
            test_heterogeneous_prefix_batch(
                verbose=args.verbose,
                backend=be,
            )

    elif args.cascade_chunk:
        # Cascade Current Chunk 测试：K/V 只有当前 block，需要 kv_offset
        for be in ["fa2", "fa3"]:
            test_cascade_current_chunk_batch(
                verbose=args.verbose,
                backend=be,
            )
