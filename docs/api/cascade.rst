.. _apicascade:

flashinfer.cascade
==================

Cascade Attention 是一种内存高效的注意力机制，通过分层 KV 缓存来减少内存占用。它特别适用于具有共享前缀的批量请求场景。

主要特性：
- 分层 KV 缓存管理
- 支持共享前缀的批量请求
- 高效的注意力状态合并
- 内存高效的推理

.. currentmodule:: flashinfer.cascade

.. _api-merge-states:

合并注意力状态
------------

合并多个注意力状态，用于层级注意力计算。

.. autofunction:: merge_state

.. autofunction:: merge_state_in_place

.. autofunction:: merge_states

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_heads = 32
    head_dim = 128

    # 准备两个注意力状态
    o1 = torch.randn(batch_size, num_heads, head_dim).half().to("cuda:0")
    m1 = torch.randn(batch_size, num_heads).float().to("cuda:0")
    d1 = torch.randn(batch_size, num_heads).float().to("cuda:0")

    o2 = torch.randn(batch_size, num_heads, head_dim).half().to("cuda:0")
    m2 = torch.randn(batch_size, num_heads).float().to("cuda:0")
    d2 = torch.randn(batch_size, num_heads).float().to("cuda:0")

    # 合并状态（创建新张量）
    o_merged, m_merged, d_merged = flashinfer.merge_state(
        o1, m1, d1, o2, m2, d2
    )

    # 原地合并（修改第一个状态）
    flashinfer.merge_state_in_place(o1, m1, d1, o2, m2, d2)

.. _api-cascade-attention:

Cascade 注意力
-------------

Cascade 注意力包装器类
---------------------

.. autoclass:: MultiLevelCascadeAttentionWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    num_layers = 32
    num_qo_heads = 64
    num_kv_heads = 8
    head_dim = 128
    page_size = 16
    max_num_pages = 256

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # 创建多级 Cascade 注意力包装器
    cascade_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 设置层级配置并执行 Cascade 注意力计算
    # ... 配置和执行代码 ...

共享前缀批量 Decode
------------------

支持共享前缀的批量 decode 注意力。

.. autoclass:: BatchDecodeWithSharedPrefixPagedKVCacheWrapper
    :members:

    .. automethod:: __init__

共享前缀批量 Prefill
-------------------

支持共享前缀的批量 prefill 注意力。

.. autoclass:: BatchPrefillWithSharedPrefixPagedKVCacheWrapper
    :members:

    .. automethod:: __init__

API 参考
--------

.. autosummary::
   :toctree: ../generated

   merge_state
   merge_state_in_place
   merge_states
