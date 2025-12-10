.. _apiattention:

FlashInfer Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlashInfer 提供了高效的注意力计算内核，支持单请求和批量请求的 decode、prefill 和 append 操作。这些内核针对 LLM 推理服务进行了优化，支持分页 KV 缓存、RoPE 位置编码、滑动窗口注意力等多种特性。

flashinfer.decode
=================

单请求和批量请求的 decode 注意力计算内核。Decode 阶段是 LLM 推理中生成新 token 的过程，通常处理单个 query token 与整个 KV 缓存的注意力计算。

.. currentmodule:: flashinfer.decode

Single Request Decoding
-----------------------

单请求 decode 注意力计算，适用于单个请求的 token 生成。

.. autofunction:: single_decode_with_kv_cache

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # 设置参数
    kv_len = 2048
    num_kv_heads = 32
    head_dim = 128
    num_qo_heads = 32

    # 准备 KV 缓存
    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")

    # 准备 query（单个 token）
    q = torch.randn(num_qo_heads, head_dim).half().to("cuda:0")

    # 执行 decode 注意力计算（无 RoPE）
    o = flashinfer.single_decode_with_kv_cache(q, k, v)

    # 执行 decode 注意力计算（带 LLaMA 风格 RoPE）
    o_rope = flashinfer.single_decode_with_kv_cache(
        q, k, v, pos_encoding_mode="ROPE_LLAMA"
    )

    print(f"输出形状: {o.shape}")  # torch.Size([32, 128])

Batch Decoding
--------------

批量 decode 注意力计算，支持使用 CUDNN 或 TensorRT-LLM 后端。

.. autofunction:: cudnn_batch_decode_with_kv_cache

.. autofunction:: trtllm_batch_decode_with_kv_cache

**批量 Decode 包装器类：**

.. autoclass:: BatchDecodeWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

**完整示例：**

.. code-block:: python

    import torch
    import flashinfer

    # 配置参数
    num_layers = 32
    num_qo_heads = 64
    num_kv_heads = 8
    head_dim = 128
    max_num_pages = 128
    page_size = 16
    batch_size = 7

    # 分配工作空间缓冲区（推荐 128MB）
    workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # 创建 decode 包装器
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 准备分页 KV 缓存索引
    kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    kv_page_indptr = torch.tensor(
        [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = torch.tensor(
        [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    )

    # 准备 KV 缓存（每个层）
    kv_cache_at_layer = [
        torch.randn(
            max_num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=torch.float16, device="cuda:0"
        ) for _ in range(num_layers)
    ]

    # 规划计算（只需执行一次，可跨层复用）
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16
    )

    # 对每个层执行 decode 注意力计算
    outputs = []
    for i in range(num_layers):
        q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
        kv_cache = kv_cache_at_layer[i]
        o = decode_wrapper.run(q, kv_cache)
        outputs.append(o)

    print(f"输出形状: {outputs[0].shape}")  # torch.Size([7, 64, 128])

**CUDA Graph 支持：**

.. autoclass:: CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    :members:

    .. automethod:: __init__

CUDA Graph 包装器允许捕获和重放计算图，减少内核启动开销，适用于固定 batch size 的场景。

XQA (eXtreme Query Attention)
------------------------------

XQA 是 FlashInfer 提供的高性能批量 decode 注意力内核，针对大规模批量推理进行了优化。

.. currentmodule:: flashinfer.xqa

.. autofunction:: xqa

.. autofunction:: xqa_mla

**XQA 示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 32
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    max_seq_len = 2048

    # 准备输入
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    k_cache = torch.randn(128, 2, page_size, num_kv_heads, head_dim).half().to("cuda:0")
    v_cache = k_cache.clone()
    page_table = torch.randint(0, 128, (batch_size, max_seq_len // page_size + 1)).int().to("cuda:0")
    seq_lens = torch.randint(100, max_seq_len, (batch_size,)).int().to("cuda:0")

    # 执行 XQA
    output = flashinfer.xqa(
        q, k_cache, v_cache, page_table, seq_lens,
        head_dim=head_dim, page_size=page_size
    )

flashinfer.prefill
==================

Prefill 和 append 注意力计算内核。Prefill 阶段处理输入序列的初始注意力计算，append 阶段处理新 token 添加到现有序列的情况。

.. currentmodule:: flashinfer.prefill

Single Request Prefill/Append Attention
---------------------------------------

单请求 prefill/append 注意力计算。

.. autofunction:: single_prefill_with_kv_cache

.. autofunction:: single_prefill_with_kv_cache_return_lse

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    kv_len = 2048
    num_kv_heads = 32
    head_dim = 128
    num_qo_heads = 32

    k = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(kv_len, num_kv_heads, head_dim).half().to("cuda:0")

    # Prefill 注意力（无因果掩码）
    qo_len = 2048
    q = torch.randn(qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=False)

    # Append 注意力（带因果掩码）
    append_qo_len = 128
    q_append = torch.randn(append_qo_len, num_qo_heads, head_dim).half().to("cuda:0")
    o_append = flashinfer.single_prefill_with_kv_cache(
        q_append, k, v, causal=True, pos_encoding_mode="ROPE_LLAMA"
    )

    # 返回 LSE（log-sum-exp）用于后续计算
    o_with_lse, lse = flashinfer.single_prefill_with_kv_cache_return_lse(
        q, k, v, causal=True
    )

Batch Prefill/Append Attention
------------------------------

批量 prefill/append 注意力计算。

.. autofunction:: cudnn_batch_prefill_with_kv_cache

.. autofunction:: trtllm_batch_context_with_kv_cache

**批量 Prefill 包装器类：**

.. autoclass:: BatchPrefillWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    max_num_pages = 256

    workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 准备分页 KV 缓存
    kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    kv_page_indptr = torch.tensor(
        [0, 32, 64, 96, 128, 160, 192, 224, 256],
        dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = torch.tensor(
        [16, 16, 16, 16, 16, 16, 16, 16],
        dtype=torch.int32, device="cuda:0"
    )

    # 准备 query 和 KV 缓存
    qo_indptr = torch.tensor(
        [0, 64, 128, 192, 256, 320, 384, 448, 512],
        dtype=torch.int32, device="cuda:0"
    )
    q = torch.randn(512, num_qo_heads, head_dim).half().to("cuda:0")
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda:0"
    )

    # 规划并执行
    prefill_wrapper.plan(
        qo_indptr,
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
        data_type=torch.float16
    )

    o = prefill_wrapper.run(q, kv_cache)
    print(f"输出形状: {o.shape}")  # torch.Size([512, 32, 128])

**Ragged KV Cache 支持：**

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

Ragged KV Cache 包装器支持非分页的 ragged KV 缓存格式，适用于某些特定的推理框架。

flashinfer.attention
====================

批量注意力计算的统一接口。

.. currentmodule:: flashinfer.attention

.. autoclass:: BatchAttention
    :members:
    :exclude-members: begin_forward, end_forward, forward

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_attention = flashinfer.BatchAttention(kv_layout="NHD", device="cuda:0")

    # 准备输入
    qo_indptr = torch.tensor([0, 64, 128, 192], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 100, 200, 300], dtype=torch.int32, device="cuda:0")
    kv_indices = torch.arange(300).int().to("cuda:0")
    kv_len_arr = torch.tensor([100, 100, 100], dtype=torch.int32, device="cuda:0")

    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16

    # 规划
    batch_attention.plan(
        qo_indptr, kv_indptr, kv_indices, kv_len_arr,
        num_qo_heads, num_kv_heads, head_dim, head_dim,
        page_size, causal=True
    )

    # 执行
    q = torch.randn(192, num_qo_heads, head_dim).half().to("cuda:0")
    k = torch.randn(300, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(300, num_kv_heads, head_dim).half().to("cuda:0")

    o = batch_attention.forward(q, k, v)

**Attention Sink 支持：**

.. autoclass:: BatchAttentionWithAttentionSinkWrapper
    :members:

    .. automethod:: __init__

Attention Sink 包装器支持注意力汇聚机制，用于处理长序列的注意力计算。

flashinfer.mla
==============

MLA (Multi-head Latent Attention) 是 DeepSeek 系列模型提出的注意力机制（
`DeepSeek-V2 <https://arxiv.org/abs/2405.04434>`_, `DeepSeek-V3 <https://arxiv.org/abs/2412.19437>`_,
和 `DeepSeek-R1 <https://arxiv.org/abs/2501.12948>`_）。

.. currentmodule:: flashinfer.mla

PageAttention for MLA
---------------------

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    max_num_pages = 256

    workspace_buffer = torch.zeros(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    mla_wrapper = flashinfer.BatchMLAPagedAttentionWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 准备输入（类似 BatchDecodeWithPagedKVCacheWrapper）
    # ... 设置分页 KV 缓存和查询 ...

    # 执行 MLA 注意力计算
    o = mla_wrapper.forward(q, kv_cache, page_table, seq_lens)
