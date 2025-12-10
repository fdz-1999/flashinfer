.. _apirope:

flashinfer.rope
===============

FlashInfer 提供了高效的旋转位置编码（RoPE）内核，支持标准 RoPE 和 LLaMA 3.1 风格的 RoPE。RoPE 是一种相对位置编码方法，广泛应用于现代 LLM 中。

主要特性：
- 支持标准 RoPE 和 LLaMA 3.1 RoPE
- 支持批量处理和位置 ID 映射
- 支持预计算 cos/sin 缓存以加速计算
- 支持原地操作以减少内存占用
- 支持 FP8 量化的 RoPE 计算

.. currentmodule:: flashinfer.rope

标准 RoPE
---------

标准旋转位置编码，适用于大多数基于 RoPE 的模型。

.. autofunction:: apply_rope

.. autofunction:: apply_rope_inplace

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_heads = 32
    head_dim = 128
    seq_len = 2048

    # 准备输入
    q = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    k = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")

    # 应用 RoPE（创建新张量）
    q_rope, k_rope = flashinfer.apply_rope(
        q, k, rotary_dim=head_dim, rope_scale=1.0, rope_theta=10000.0
    )

    # 原地应用 RoPE（修改输入张量）
    flashinfer.apply_rope_inplace(
        q, k, rotary_dim=head_dim, rope_scale=1.0, rope_theta=10000.0
    )

使用位置 ID
-----------

当需要为不同的 token 指定不同的位置时，可以使用位置 ID 版本。

.. autofunction:: apply_rope_pos_ids

.. autofunction:: apply_rope_pos_ids_inplace

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_heads = 32
    head_dim = 128
    seq_len = 2048

    q = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    k = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")

    # 定义位置 ID（例如，某些 token 的位置可能不连续）
    pos_ids = torch.arange(seq_len).int().to("cuda:0")

    # 使用位置 ID 应用 RoPE
    q_rope, k_rope = flashinfer.apply_rope_pos_ids(
        q, k, pos_ids, rotary_dim=head_dim, rope_scale=1.0, rope_theta=10000.0
    )

使用预计算缓存
-------------

对于重复计算，可以预计算 cos/sin 缓存以提高性能。

.. autofunction:: apply_rope_with_cos_sin_cache

.. autofunction:: apply_rope_with_cos_sin_cache_inplace

**示例：**

.. code-block:: python

    import torch
    import flashinfer
    import math

    batch_size = 8
    num_heads = 32
    head_dim = 128
    max_seq_len = 4096
    rope_theta = 10000.0

    # 预计算 cos/sin 缓存
    rotary_dim = head_dim
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    t = torch.arange(max_seq_len).float().unsqueeze(1) * inv_freq.unsqueeze(0)
    cos_cache = torch.cos(t).half().to("cuda:0")
    sin_cache = torch.sin(t).half().to("cuda:0")

    # 使用缓存应用 RoPE
    q = torch.randn(2048, num_heads, head_dim).half().to("cuda:0")
    k = torch.randn(2048, num_heads, head_dim).half().to("cuda:0")
    pos_ids = torch.arange(2048).int().to("cuda:0")

    q_rope, k_rope = flashinfer.apply_rope_with_cos_sin_cache(
        q, k, pos_ids, cos_cache, sin_cache, rotary_dim=head_dim
    )

LLaMA 3.1 RoPE
--------------

LLaMA 3.1 引入了改进的 RoPE 实现，支持更长的上下文长度。

.. autofunction:: apply_llama31_rope

.. autofunction:: apply_llama31_rope_inplace

.. autofunction:: apply_llama31_rope_pos_ids

.. autofunction:: apply_llama31_rope_pos_ids_inplace

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_heads = 32
    head_dim = 128
    seq_len = 2048

    q = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")
    k = torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0")

    # 应用 LLaMA 3.1 RoPE
    q_rope, k_rope = flashinfer.apply_llama31_rope(
        q, k,
        rotary_dim=head_dim,
        rope_scale=1.0,
        rope_theta=500000.0,  # LLaMA 3.1 使用更大的 theta
        low_freq_factor=1.0,
        high_freq_factor=1.0,
        old_context_len=2048
    )

量化 RoPE
---------

FlashInfer 还支持 FP8 量化的 RoPE 计算，用于压缩 KV 缓存。

.. autofunction:: rope_quantize_fp8

.. autofunction:: rope_quantize_fp8_append_paged_kv_cache

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # FP8 量化的 RoPE 计算
    q = torch.randn(2048, 32, 128).half().to("cuda:0")
    k = torch.randn(2048, 32, 128).half().to("cuda:0")

    # 量化并应用 RoPE
    q_rope_fp8, k_rope_fp8 = flashinfer.rope_quantize_fp8(
        q, k, rotary_dim=128, rope_scale=1.0, rope_theta=10000.0
    )

批量处理示例
------------

以下是一个完整的批量 RoPE 应用示例：

.. code-block:: python

    import torch
    import flashinfer

    def apply_rope_batch(queries, keys, seq_lens, rope_theta=10000.0):
        """为批量请求应用 RoPE"""
        batch_size = len(seq_lens)
        num_heads = queries.shape[1]
        head_dim = queries.shape[2]
        
        # 构建 indptr
        indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=queries.device)
        for i, seq_len in enumerate(seq_lens):
            indptr[i + 1] = indptr[i] + seq_len
        
        # 构建 offsets（位置偏移）
        offsets = torch.zeros(batch_size, dtype=torch.int32, device=queries.device)
        
        # 应用 RoPE
        q_rope, k_rope = flashinfer.apply_rope(
            queries, keys,
            indptr=indptr,
            offsets=offsets,
            rotary_dim=head_dim,
            rope_scale=1.0,
            rope_theta=rope_theta
        )
        
        return q_rope, k_rope

    # 使用示例
    batch_size = 4
    seq_lens = [128, 256, 512, 1024]
    num_heads = 32
    head_dim = 128
    
    queries = []
    keys = []
    for seq_len in seq_lens:
        queries.append(torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0"))
        keys.append(torch.randn(seq_len, num_heads, head_dim).half().to("cuda:0"))
    
    q_batch = torch.cat(queries, dim=0)
    k_batch = torch.cat(keys, dim=0)
    
    q_rope, k_rope = apply_rope_batch(q_batch, k_batch, seq_lens)

API 参考
--------

.. autosummary::
    :toctree: ../generated

    apply_rope_inplace
    apply_llama31_rope_inplace
    apply_rope
    apply_llama31_rope
    apply_rope_pos_ids
    apply_rope_pos_ids_inplace
    apply_llama31_rope_pos_ids
    apply_llama31_rope_pos_ids_inplace
    apply_rope_with_cos_sin_cache
    apply_rope_with_cos_sin_cache_inplace
    rope_quantize_fp8
    rope_quantize_fp8_append_paged_kv_cache
