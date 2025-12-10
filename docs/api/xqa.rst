.. _apixqa:

flashinfer.xqa
==============

XQA (eXtreme Query Attention) 是 FlashInfer 提供的高性能批量 decode 注意力内核，专门针对大规模批量推理场景进行了优化。XQA 内核可以高效处理大量并发请求的 decode 阶段计算。

主要特性：
- 高性能批量 decode 注意力计算
- 支持分页 KV 缓存
- 支持滑动窗口注意力
- 支持 FP8 量化
- 支持注意力汇聚（Attention Sink）
- 支持推测性解码

.. currentmodule:: flashinfer.xqa

标准 XQA
--------

标准 XQA 内核，适用于大多数批量 decode 场景。

.. autofunction:: xqa

**参数说明：**

- ``q``: Query 张量，形状为 ``(batch_size, num_qo_heads, head_dim)``
- ``k_cache``: K 缓存，形状为 ``(max_num_pages, page_size, num_kv_heads, head_dim)``
- ``v_cache``: V 缓存，形状与 k_cache 相同
- ``page_table``: 页面表，形状为 ``(batch_size, max_pages_per_seq)``
- ``seq_lens``: 每个请求的序列长度，形状为 ``(batch_size,)``
- ``head_dim``: 注意力头维度
- ``page_size``: 页面大小（通常为 16）
- ``num_kv_heads``: KV 注意力头数量
- ``pos_encoding_mode``: 位置编码模式（"NONE", "ROPE_LLAMA" 等）
- ``sliding_window_size``: 滑动窗口大小（-1 表示不使用滑动窗口）

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 32
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    max_num_pages = 512
    max_seq_len = 4096

    # 准备输入
    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    
    # 准备分页 KV 缓存
    k_cache = torch.randn(
        max_num_pages, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda:0"
    )
    v_cache = k_cache.clone()
    
    # 准备页面表（每个请求的页面索引）
    max_pages_per_seq = max_seq_len // page_size + 1
    page_table = torch.randint(
        0, max_num_pages, (batch_size, max_pages_per_seq)
    ).int().to("cuda:0")
    
    # 准备序列长度
    seq_lens = torch.randint(100, max_seq_len, (batch_size,)).int().to("cuda:0")

    # 执行 XQA
    output = flashinfer.xqa(
        q, k_cache, v_cache, page_table, seq_lens,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        pos_encoding_mode="NONE"
    )
    
    print(f"输出形状: {output.shape}")  # torch.Size([32, 32, 128])

带 RoPE 的 XQA
-------------

使用旋转位置编码的 XQA：

.. code-block:: python

    import torch
    import flashinfer

    # 使用 LLaMA 风格的 RoPE
    output = flashinfer.xqa(
        q, k_cache, v_cache, page_table, seq_lens,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        pos_encoding_mode="ROPE_LLAMA",
        rope_scale=1.0,
        rope_theta=10000.0
    )

带滑动窗口的 XQA
---------------

使用滑动窗口注意力限制注意力范围：

.. code-block:: python

    import torch
    import flashinfer

    sliding_window_size = 2048  # 滑动窗口大小
    
    output = flashinfer.xqa(
        q, k_cache, v_cache, page_table, seq_lens,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        sliding_window_size=sliding_window_size
    )

带注意力汇聚的 XQA
----------------

使用注意力汇聚机制处理长序列：

.. code-block:: python

    import torch
    import flashinfer

    # 准备注意力汇聚 token
    num_sink_tokens = 4
    sinks = torch.randn(
        num_sink_tokens, num_kv_heads, head_dim
    ).half().to("cuda:0")
    
    output = flashinfer.xqa(
        q, k_cache, v_cache, page_table, seq_lens,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        sinks=sinks
    )

MLA XQA
-------

支持 MLA (Multi-head Latent Attention) 的 XQA 内核。

.. autofunction:: xqa_mla

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 32
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    max_num_pages = 512

    # MLA 使用压缩的 KV 缓存格式
    mla_kv_cache = torch.randn(
        max_num_pages, page_size, num_kv_heads, head_dim * 2,
        dtype=torch.float16, device="cuda:0"
    )

    q = torch.randn(batch_size, num_qo_heads, head_dim).half().to("cuda:0")
    page_table = torch.randint(0, max_num_pages, (batch_size, 32)).int().to("cuda:0")
    seq_lens = torch.randint(100, 500, (batch_size,)).int().to("cuda:0")

    # 执行 MLA XQA
    output = flashinfer.xqa_mla(
        q, mla_kv_cache, page_table, seq_lens,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_heads=num_kv_heads
    )

完整批量推理示例
--------------

以下是一个完整的批量推理循环示例：

.. code-block:: python

    import torch
    import flashinfer

    class BatchInferenceEngine:
        def __init__(self, num_layers, num_qo_heads, num_kv_heads, head_dim, page_size):
            self.num_layers = num_layers
            self.num_qo_heads = num_qo_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.page_size = page_size
            
            # 为每层分配 KV 缓存
            self.kv_caches = [
                torch.zeros(512, page_size, num_kv_heads, head_dim,
                           dtype=torch.float16, device="cuda:0")
                for _ in range(num_layers)
            ]
        
        def decode_step(self, layer_idx, q, page_table, seq_lens):
            """执行单层 decode"""
            k_cache = self.kv_caches[layer_idx]
            v_cache = k_cache  # 简化示例
            
            output = flashinfer.xqa(
                q, k_cache, v_cache, page_table, seq_lens,
                head_dim=self.head_dim,
                page_size=self.page_size,
                num_kv_heads=self.num_kv_heads,
                pos_encoding_mode="ROPE_LLAMA"
            )
            
            return output
        
        def generate_step(self, queries, page_tables, seq_lens):
            """生成一个 token"""
            batch_size = queries.shape[0]
            
            # 对每一层执行 decode
            hidden_states = queries
            for layer_idx in range(self.num_layers):
                # 线性投影（简化，实际应用中需要完整的 Transformer 层）
                q = hidden_states  # 简化
                
                # XQA 注意力
                attn_output = self.decode_step(layer_idx, q, page_tables, seq_lens)
                
                # 更新隐藏状态（简化）
                hidden_states = attn_output
            
            return hidden_states

    # 使用示例
    engine = BatchInferenceEngine(
        num_layers=32, num_qo_heads=32, num_kv_heads=8,
        head_dim=128, page_size=16
    )
    
    batch_size = 16
    q = torch.randn(batch_size, 32, 128).half().to("cuda:0")
    page_table = torch.randint(0, 512, (batch_size, 32)).int().to("cuda:0")
    seq_lens = torch.randint(100, 500, (batch_size,)).int().to("cuda:0")
    
    output = engine.generate_step(q, page_table, seq_lens)

性能优化提示
------------

1. **批量大小**：XQA 在较大批量时性能更好，建议批量大小 >= 16
2. **页面大小**：使用标准的页面大小（16）以获得最佳性能
3. **内存对齐**：确保 KV 缓存的内存布局正确对齐
4. **序列长度**：对于非常长的序列，考虑使用滑动窗口或注意力汇聚

API 参考
--------

.. autosummary::
    :toctree: ../generated

    xqa
    xqa_mla
