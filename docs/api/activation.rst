.. _apiactivation:

flashinfer.activation
=====================

FlashInfer 提供了高效的激活函数内核，专门用于 Transformer MLP 层的 up/gate 结构。这些内核融合了激活函数和乘法操作，减少了内存访问和内核启动开销。

主要特性：
- 融合的激活和乘法操作
- 支持 SiLU、GELU 等常用激活函数
- 支持量化版本的激活函数
- 针对 LLM 推理优化

.. currentmodule:: flashinfer.activation

SiLU 激活
---------

SiLU (Sigmoid Linear Unit) 激活函数，也称为 Swish，广泛用于现代 LLM。

.. autofunction:: silu_and_mul

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096

    # 输入形状为 (..., 2 * hidden_size)
    # 前一半是 gate，后一半是 up
    input_tensor = torch.randn(batch_size, 2 * hidden_size).half().to("cuda:0")

    # 融合操作：SiLU(gate) * up
    output = flashinfer.silu_and_mul(input_tensor)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

    # 原地操作
    output = torch.empty(batch_size, hidden_size, dtype=torch.float16, device="cuda:0")
    flashinfer.silu_and_mul(input_tensor, out=output)

GELU 激活
---------

GELU (Gaussian Error Linear Unit) 激活函数。

.. autofunction:: gelu_and_mul

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096

    input_tensor = torch.randn(batch_size, 2 * hidden_size).half().to("cuda:0")

    # 融合操作：GELU(gate) * up
    output = flashinfer.gelu_and_mul(input_tensor)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

GELU Tanh 激活
-------------

使用 tanh 近似的 GELU 激活函数。

.. autofunction:: gelu_tanh_and_mul

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096

    input_tensor = torch.randn(batch_size, 2 * hidden_size).half().to("cuda:0")

    # 融合操作：GELU_tanh(gate) * up
    output = flashinfer.gelu_tanh_and_mul(input_tensor)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

量化激活
--------

支持量化版本的激活函数，用于 MoE 模型的专家量化。

.. autofunction:: silu_and_mul_scaled_nvfp4_experts_quantize

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096
    num_experts = 8

    input_tensor = torch.randn(batch_size, 2 * hidden_size).half().to("cuda:0")
    expert_scales = torch.randn(num_experts, hidden_size).half().to("cuda:0")
    expert_indices = torch.randint(0, num_experts, (batch_size,)).int().to("cuda:0")

    # 量化版本的 SiLU 激活
    output = flashinfer.silu_and_mul_scaled_nvfp4_experts_quantize(
        input_tensor, expert_scales, expert_indices
    )

完整 MLP 层示例
--------------

以下是一个完整的 MLP 层示例：

.. code-block:: python

    import torch
    import flashinfer

    class MLPLayer:
        def __init__(self, hidden_size, intermediate_size):
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            
            # 初始化权重（实际应用中从模型加载）
            self.gate_proj = torch.randn(hidden_size, intermediate_size).half().to("cuda:0")
            self.up_proj = torch.randn(hidden_size, intermediate_size).half().to("cuda:0")
            self.down_proj = torch.randn(intermediate_size, hidden_size).half().to("cuda:0")
            
        def forward(self, x):
            """前向传播"""
            # Gate 和 Up 投影
            gate = torch.matmul(x, self.gate_proj)  # (batch, intermediate_size)
            up = torch.matmul(x, self.up_proj)      # (batch, intermediate_size)
            
            # 拼接 gate 和 up
            gate_up = torch.cat([gate, up], dim=-1)  # (batch, 2 * intermediate_size)
            
            # 融合激活和乘法
            activated = flashinfer.silu_and_mul(gate_up)  # (batch, intermediate_size)
            
            # Down 投影
            output = torch.matmul(activated, self.down_proj)  # (batch, hidden_size)
            
            return output

    # 使用示例
    mlp = MLPLayer(hidden_size=4096, intermediate_size=11008)
    x = torch.randn(8, 4096).half().to("cuda:0")
    output = mlp.forward(x)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

性能优化提示
------------

1. **使用融合操作**：始终使用融合的激活函数而不是分别执行激活和乘法
2. **内存对齐**：确保输入张量的最后一个维度是 16 字节对齐的
3. **批量处理**：对于多个序列，使用批量处理可以提高 GPU 利用率

API 参考
--------

.. autosummary::
    :toctree: ../generated

    silu_and_mul
    gelu_tanh_and_mul
    gelu_and_mul
    silu_and_mul_scaled_nvfp4_experts_quantize
