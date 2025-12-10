.. _apigemm:

flashinfer.gemm
===============

FlashInfer 提供了高效的 GEMM（通用矩阵乘法）操作，支持多种精度和格式，包括 FP4、FP8 以及分组 GEMM。这些内核针对 LLM 推理中的线性层计算进行了优化。

主要特性：
- 支持 FP4 和 FP8 量化 GEMM
- 支持分组 GEMM（用于 MoE 模型）
- 支持混合精度计算
- 针对不同 GPU 架构优化（Ampere、Hopper、Blackwell）

.. currentmodule:: flashinfer.gemm

FP4 GEMM
--------

FP4 量化矩阵乘法，用于压缩模型权重。

.. autofunction:: mm_fp4

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096
    intermediate_size = 11008

    # 准备输入
    x = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    
    # FP4 量化权重（实际应用中从模型加载）
    weight_fp4 = torch.randint(0, 255, (hidden_size, intermediate_size), 
                                dtype=torch.uint8, device="cuda:0")
    weight_scale = torch.randn(intermediate_size).half().to("cuda:0")

    # FP4 GEMM
    output = flashinfer.mm_fp4(x, weight_fp4, weight_scale)

FP8 GEMM
--------

FP8 量化矩阵乘法，支持批量操作和分组操作。

.. autofunction:: bmm_fp8

.. autofunction:: mm_fp8

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096
    intermediate_size = 11008

    # 批量 FP8 GEMM
    x = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    weight_fp8 = torch.randn(hidden_size, intermediate_size).half().to("cuda:0")
    weight_scale = torch.randn(intermediate_size).half().to("cuda:0")

    output = flashinfer.bmm_fp8(x, weight_fp8, weight_scale)

分组 GEMM
---------

分组 GEMM 用于 MoE 模型，可以高效处理多个专家的权重。

.. autoclass:: SegmentGEMMWrapper
    :members:
    :exclude-members: forward

    .. automethod:: __init__

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096
    intermediate_size = 11008
    num_experts = 8

    # 创建分组 GEMM 包装器
    gemm_wrapper = flashinfer.SegmentGEMMWrapper(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device="cuda:0"
    )

    # 准备输入
    x = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    expert_weights = torch.randn(
        num_experts, hidden_size, intermediate_size
    ).half().to("cuda:0")
    expert_indices = torch.randint(0, num_experts, (batch_size,)).int().to("cuda:0")

    # 执行分组 GEMM
    output = gemm_wrapper.forward(x, expert_weights, expert_indices)

TGV GEMM
--------

Tensor-Gather-Vector GEMM，针对特定架构优化。

.. autofunction:: tgv_gemm_sm100

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # TGV GEMM 用于 SM100+ 架构
    a = torch.randn(1024, 4096).half().to("cuda:0")
    b = torch.randn(4096, 11008).half().to("cuda:0")
    
    output = flashinfer.tgv_gemm_sm100(a, b)

完整线性层示例
--------------

以下是一个使用量化 GEMM 的完整线性层示例：

.. code-block:: python

    import torch
    import flashinfer

    class QuantizedLinearLayer:
        def __init__(self, in_features, out_features, use_fp8=True):
            self.in_features = in_features
            self.out_features = out_features
            self.use_fp8 = use_fp8
            
            if use_fp8:
                # FP8 权重（实际应用中需要量化）
                self.weight = torch.randn(in_features, out_features).half().to("cuda:0")
                self.weight_scale = torch.ones(out_features).half().to("cuda:0")
            else:
                # FP4 权重
                self.weight = torch.randint(0, 255, (in_features, out_features),
                                          dtype=torch.uint8, device="cuda:0")
                self.weight_scale = torch.ones(out_features).half().to("cuda:0")
        
        def forward(self, x):
            """前向传播"""
            if self.use_fp8:
                return flashinfer.bmm_fp8(x, self.weight, self.weight_scale)
            else:
                return flashinfer.mm_fp4(x, self.weight, self.weight_scale)

    # 使用示例
    layer = QuantizedLinearLayer(4096, 11008, use_fp8=True)
    x = torch.randn(8, 4096).half().to("cuda:0")
    output = layer.forward(x)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 11008])

性能优化提示
------------

1. **选择合适的精度**：FP8 通常比 FP4 更快，但占用更多内存
2. **批量处理**：尽可能使用批量操作以提高 GPU 利用率
3. **权重布局**：确保权重采用正确的布局以匹配内核要求

API 参考
--------

.. autosummary::
    :toctree: ../generated

    mm_fp4
    mm_fp8
    bmm_fp8
    tgv_gemm_sm100
