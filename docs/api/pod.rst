.. _apipod:

flashinfer.pod
==============

POD (Parallel Output Decode) Attention 是一种高效的注意力机制，专为批量 decode 场景设计。POD 注意力通过并行化输出计算来提高性能。

主要特性：
- 高效的批量 decode 注意力计算
- 支持分页 KV 缓存
- 支持 RoPE 位置编码
- 针对大规模批量推理优化

.. currentmodule:: flashinfer.pod

单请求 POD
----------

单请求 POD 注意力计算。

.. autoclass:: PODWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    num_layers = 32
    num_qo_heads = 64
    num_kv_heads = 8
    head_dim = 128
    max_num_pages = 128
    page_size = 16

    # 分配工作空间缓冲区
    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # 创建 POD 包装器
    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    batch_size = 7
    kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    kv_page_indptr = torch.tensor(
        [0, 17, 29, 44, 48, 66, 100, 128], dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = torch.tensor(
        [1, 7, 14, 4, 3, 1, 16], dtype=torch.int32, device="cuda:0"
    )

    # 准备 KV 缓存
    kv_cache_at_layer = [
        torch.randn(
            max_num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=torch.float16, device="cuda:0"
        ) for _ in range(num_layers)
    ]

    # 规划计算
    pod_wrapper.plan(
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

    # 对每层执行 POD 注意力
    outputs = []
    for i in range(num_layers):
        q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
        kv_cache = kv_cache_at_layer[i]
        o = pod_wrapper.forward(q, kv_cache)
        outputs.append(o)

    print(f"输出形状: {outputs[0].shape}")  # torch.Size([7, 64, 128])

批量 POD
--------

批量 POD 注意力计算，支持多个请求的并行处理。

.. autoclass:: BatchPODWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    num_layers = 32
    num_qo_heads = 64
    num_kv_heads = 8
    head_dim = 128
    max_num_pages = 256
    page_size = 16
    batch_size = 16

    # 分配工作空间缓冲区
    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # 创建批量 POD 包装器
    batch_pod_wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 准备批量页面表
    kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
    kv_page_indptr = torch.tensor(
        [0] + [i * 16 for i in range(1, batch_size + 1)],
        dtype=torch.int32, device="cuda:0"
    )
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

    # 准备 KV 缓存
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda:0"
    )

    # 规划计算
    batch_pod_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="ROPE_LLAMA",
        data_type=torch.float16
    )

    # 执行批量 POD 注意力
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    o = batch_pod_wrapper.forward(q, kv_cache)

    print(f"输出形状: {o.shape}")  # torch.Size([16, 64, 128])

带 RoPE 的 POD
-------------

使用旋转位置编码的 POD 注意力：

.. code-block:: python

    import torch
    import flashinfer

    # 创建 POD 包装器
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    pod_wrapper = flashinfer.PODWithPagedKVCacheWrapper(workspace_buffer)

    # 规划时指定 RoPE
    pod_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="ROPE_LLAMA",
        rope_scale=1.0,
        rope_theta=10000.0,
        data_type=torch.float16
    )

    # 执行时 RoPE 会自动应用
    o = pod_wrapper.forward(q, kv_cache)

完整推理循环示例
--------------

以下是一个使用 POD 注意力的完整推理循环示例：

.. code-block:: python

    import torch
    import flashinfer

    class PODInferenceEngine:
        def __init__(self, num_layers, num_qo_heads, num_kv_heads, head_dim, page_size):
            self.num_layers = num_layers
            self.num_qo_heads = num_qo_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.page_size = page_size
            
            # 分配工作空间
            self.workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
            )
            
            # 为每层创建 POD 包装器
            self.pod_wrappers = [
                flashinfer.PODWithPagedKVCacheWrapper(self.workspace_buffer)
                for _ in range(num_layers)
            ]
            
            # KV 缓存
            self.kv_caches = []
        
        def setup_batch(self, batch_size, max_num_pages):
            """设置批量请求"""
            # 准备页面表
            self.kv_page_indices = torch.arange(max_num_pages).int().to("cuda:0")
            self.kv_page_indptr = torch.tensor(
                [0] + [i * 16 for i in range(1, batch_size + 1)],
                dtype=torch.int32, device="cuda:0"
            )
            self.kv_last_page_len = torch.ones(
                batch_size, dtype=torch.int32, device="cuda:0"
            )
            
            # 为每层分配 KV 缓存
            self.kv_caches = [
                torch.randn(
                    max_num_pages, 2, self.page_size, self.num_kv_heads, self.head_dim,
                    dtype=torch.float16, device="cuda:0"
                ) for _ in range(self.num_layers)
            ]
            
            # 规划每层的 POD 计算
            for pod_wrapper in self.pod_wrappers:
                pod_wrapper.plan(
                    self.kv_page_indptr,
                    self.kv_page_indices,
                    self.kv_last_page_len,
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                    pos_encoding_mode="ROPE_LLAMA",
                    data_type=torch.float16
                )
        
        def decode_step(self, q):
            """执行一个 decode 步骤"""
            hidden_states = q
            for layer_idx in range(self.num_layers):
                # POD 注意力
                attn_output = self.pod_wrappers[layer_idx].forward(
                    hidden_states, self.kv_caches[layer_idx]
                )
                # 更新隐藏状态（简化，实际应用中需要完整的 Transformer 层）
                hidden_states = attn_output
            return hidden_states

    # 使用示例
    engine = PODInferenceEngine(
        num_layers=32, num_qo_heads=64, num_kv_heads=8,
        head_dim=128, page_size=16
    )
    
    engine.setup_batch(batch_size=16, max_num_pages=256)
    
    # 执行 decode
    q = torch.randn(16, 64, 128).half().to("cuda:0")
    output = engine.decode_step(q)

性能优化提示
------------

1. **工作空间大小**：推荐使用 128MB 的工作空间缓冲区
2. **批量大小**：POD 在较大批量时性能更好
3. **页面大小**：使用标准的页面大小（16）以获得最佳性能
4. **复用规划**：对于固定批量大小的场景，可以复用规划结果

API 参考
--------

.. autosummary::
    :toctree: ../generated

    PODWithPagedKVCacheWrapper
    BatchPODWithPagedKVCacheWrapper
