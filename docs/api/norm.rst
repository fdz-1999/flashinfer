.. _apinorm:

flashinfer.norm
===============

FlashInfer 提供了高效的归一化层内核，包括 RMSNorm、LayerNorm 以及融合的残差连接操作。这些内核针对 LLM 推理进行了优化。

主要特性：
- 高效的 RMSNorm 和 LayerNorm 实现
- 融合的残差连接操作（减少内存访问）
- 支持 Gemma 模型的特殊归一化
- 支持 Programmatic Dependent Launch (PDL) 以提升性能

.. currentmodule:: flashinfer.norm

RMSNorm
-------

Root Mean Square Normalization，广泛用于现代 LLM 中。

.. autofunction:: rmsnorm

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096
    seq_len = 2048

    # 2D 输入（batch_size, hidden_size）
    input_2d = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    weight = torch.ones(hidden_size).half().to("cuda:0")

    output = flashinfer.rmsnorm(input_2d, weight, eps=1e-6)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

    # 3D 输入（batch_size, num_heads, head_dim）
    num_heads = 32
    head_dim = 128
    input_3d = torch.randn(batch_size, num_heads, head_dim).half().to("cuda:0")
    weight_3d = torch.ones(head_dim).half().to("cuda:0")

    output_3d = flashinfer.rmsnorm(input_3d, weight_3d, eps=1e-6)
    print(f"输出形状: {output_3d.shape}")  # torch.Size([8, 32, 128])

融合残差连接的 RMSNorm
---------------------

融合操作可以减少内存访问，提高性能。

.. autofunction:: fused_add_rmsnorm

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096

    input_tensor = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    residual = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    weight = torch.ones(hidden_size).half().to("cuda:0")

    # 融合操作：residual = input + residual，然后应用 RMSNorm
    # 注意：这会原地修改 input 和 residual
    flashinfer.fused_add_rmsnorm(
        input_tensor, residual, weight, eps=1e-6
    )

    # input_tensor 现在包含归一化后的结果
    # residual 现在包含 input + residual 的结果

LayerNorm
---------

标准 LayerNorm 实现。

.. autofunction:: layernorm

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096

    input_tensor = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    weight = torch.ones(hidden_size).half().to("cuda:0")
    bias = torch.zeros(hidden_size).half().to("cuda:0")

    output = flashinfer.layernorm(
        input_tensor, weight, bias, eps=1e-6
    )

Gemma 归一化
-----------

Gemma 模型使用的特殊归一化实现。

.. autofunction:: gemma_rmsnorm

.. autofunction:: gemma_fused_add_rmsnorm

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096

    input_tensor = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    weight = torch.ones(hidden_size).half().to("cuda:0")

    # Gemma RMSNorm
    output = flashinfer.gemma_rmsnorm(input_tensor, weight, eps=1e-6)

    # Gemma 融合残差连接
    residual = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    flashinfer.gemma_fused_add_rmsnorm(
        input_tensor, residual, weight, eps=1e-6
    )

完整 Transformer 层示例
------------------------

以下是一个完整的 Transformer 层示例，展示如何使用归一化操作：

.. code-block:: python

    import torch
    import flashinfer

    class TransformerLayer:
        def __init__(self, hidden_size, num_heads, head_dim):
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = head_dim
            
            # 初始化权重
            self.attn_norm_weight = torch.ones(hidden_size).half().to("cuda:0")
            self.ffn_norm_weight = torch.ones(hidden_size).half().to("cuda:0")
            
        def forward(self, x, attn_output, ffn_output):
            """前向传播"""
            # 注意力后的归一化（融合残差连接）
            flashinfer.fused_add_rmsnorm(
                attn_output, x, self.attn_norm_weight, eps=1e-6
            )
            # attn_output 现在包含归一化后的结果
            
            # FFN 后的归一化（融合残差连接）
            flashinfer.fused_add_rmsnorm(
                ffn_output, attn_output, self.ffn_norm_weight, eps=1e-6
            )
            # ffn_output 现在包含最终输出
            
            return ffn_output

    # 使用示例
    layer = TransformerLayer(hidden_size=4096, num_heads=32, head_dim=128)
    x = torch.randn(8, 4096).half().to("cuda:0")
    attn_out = torch.randn(8, 4096).half().to("cuda:0")
    ffn_out = torch.randn(8, 4096).half().to("cuda:0")
    
    output = layer.forward(x, attn_out, ffn_out)

性能优化提示
------------

1. **使用融合操作**：尽可能使用 `fused_add_rmsnorm` 而不是分别执行加法和归一化
2. **启用 PDL**：在支持的设备上，PDL 会自动启用以提升性能
3. **批量处理**：对于多个序列，使用批量处理可以提高 GPU 利用率

API 参考
--------

.. autosummary::
    :toctree: ../generated

    rmsnorm
    fused_add_rmsnorm
    gemma_rmsnorm
    gemma_fused_add_rmsnorm
    layernorm
