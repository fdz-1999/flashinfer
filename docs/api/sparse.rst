.. _apisparse:

flashinfer.sparse
=================

FlashInfer 提供了块稀疏注意力内核，可以显著减少长序列的注意力计算量。稀疏注意力只计算特定块的注意力，而不是所有位置对。

主要特性：
- 固定块稀疏注意力
- 可变块稀疏注意力
- 高效的稀疏模式规划
- 支持长序列注意力计算

.. currentmodule:: flashinfer.sparse

固定块稀疏注意力
---------------

固定块大小的稀疏注意力。

.. autoclass:: BlockSparseAttentionWrapper
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
    seq_len = 4096
    block_size = 64

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # 创建块稀疏注意力包装器
    sparse_wrapper = flashinfer.BlockSparseAttentionWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 准备输入
    q = torch.randn(batch_size, seq_len, num_qo_heads, head_dim).half().to("cuda:0")
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim).half().to("cuda:0")

    # 定义稀疏模式（哪些块需要计算）
    # 这里使用简单的对角线模式
    num_blocks = seq_len // block_size
    sparse_mask = torch.zeros(batch_size, num_blocks, num_blocks, dtype=torch.bool, device="cuda:0")
    for i in range(num_blocks):
        sparse_mask[:, i, i] = True  # 对角线块
        if i > 0:
            sparse_mask[:, i, i-1] = True  # 前一个块

    # 规划稀疏注意力
    sparse_wrapper.plan(
        q, k, v, sparse_mask,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )

    # 执行稀疏注意力
    o = sparse_wrapper.forward(q, k, v)

可变块稀疏注意力
---------------

支持可变块大小的稀疏注意力，提供更灵活的稀疏模式。

.. autoclass:: VariableBlockSparseAttentionWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. image:: https://raw.githubusercontent.com/flashinfer-ai/web-data/main/examples/flashinfer-variable-block-sparse.png
        :width: 600
        :alt: variable block sparse attention plan function diagram
        :align: center

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 4096

    workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # 创建可变块稀疏注意力包装器
    var_sparse_wrapper = flashinfer.VariableBlockSparseAttentionWrapper(
        workspace_buffer, kv_layout="NHD"
    )

    # 准备输入
    q = torch.randn(batch_size, seq_len, num_qo_heads, head_dim).half().to("cuda:0")
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim).half().to("cuda:0")
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim).half().to("cuda:0")

    # 定义可变块大小和稀疏模式
    # ... 设置块配置 ...

    # 规划可变块稀疏注意力
    var_sparse_wrapper.plan(
        q, k, v,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim
    )

    # 执行可变块稀疏注意力
    o = var_sparse_wrapper.forward(q, k, v)

稀疏模式转换
-----------

.. autofunction:: convert_bsr_mask_layout

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # 转换 BSR 掩码布局
    mask = torch.randn(8, 64, 64).half().to("cuda:0")
    indptr = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda:0")
    
    converted_mask = flashinfer.convert_bsr_mask_layout(mask, indptr)

完整稀疏注意力示例
------------------

以下是一个使用稀疏注意力的完整示例：

.. code-block:: python

    import torch
    import flashinfer

    class SparseAttentionLayer:
        def __init__(self, num_qo_heads, num_kv_heads, head_dim, block_size=64):
            self.num_qo_heads = num_qo_heads
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            self.block_size = block_size
            
            workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
            )
            
            self.sparse_wrapper = flashinfer.BlockSparseAttentionWrapper(
                workspace_buffer
            )
        
        def create_sparse_pattern(self, seq_len, pattern_type="local"):
            """创建稀疏模式"""
            num_blocks = seq_len // self.block_size
            mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool, device="cuda:0")
            
            if pattern_type == "local":
                # 局部注意力：每个块只关注附近的块
                window_size = 2
                for i in range(num_blocks):
                    for j in range(max(0, i - window_size), min(num_blocks, i + window_size + 1)):
                        mask[i, j] = True
            elif pattern_type == "strided":
                # 跨步注意力：每隔几个块关注一次
                stride = 4
                for i in range(num_blocks):
                    for j in range(0, num_blocks, stride):
                        mask[i, j] = True
            
            return mask
        
        def forward(self, q, k, v, sparse_pattern):
            """前向传播"""
            batch_size = q.shape[0]
            sparse_mask = sparse_pattern.unsqueeze(0).expand(batch_size, -1, -1)
            
            # 规划
            self.sparse_wrapper.plan(
                q, k, v, sparse_mask,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                block_size=self.block_size
            )
            
            # 执行
            return self.sparse_wrapper.forward(q, k, v)

    # 使用示例
    layer = SparseAttentionLayer(num_qo_heads=32, num_kv_heads=8, head_dim=128)
    
    seq_len = 4096
    q = torch.randn(8, seq_len, 32, 128).half().to("cuda:0")
    k = torch.randn(8, seq_len, 8, 128).half().to("cuda:0")
    v = torch.randn(8, seq_len, 8, 128).half().to("cuda:0")
    
    sparse_pattern = layer.create_sparse_pattern(seq_len, pattern_type="local")
    output = layer.forward(q, k, v, sparse_pattern)

性能优化提示
------------

1. **块大小**：选择合适的块大小以平衡计算效率和稀疏性
2. **稀疏模式**：根据任务特性设计合适的稀疏模式
3. **规划复用**：对于固定稀疏模式的场景，可以复用规划结果

API 参考
--------

.. autosummary::
    :toctree: ../generated

    convert_bsr_mask_layout
