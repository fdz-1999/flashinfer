.. _apifused_moe:

flashinfer.fused_moe
====================

FlashInfer 提供了融合的混合专家（MoE）操作，针对不同的后端和数据类型进行了优化。MoE 模型通过使用多个专家网络来提高模型容量，同时保持计算效率。

主要特性：
- 支持 CUTLASS 和 TensorRT-LLM 后端
- 支持 FP4 和 FP8 量化
- 支持多种路由方法
- 高效的专家权重布局转换
- 针对不同 GPU 架构优化

.. currentmodule:: flashinfer.fused_moe

类型和枚举
----------

.. autosummary::
    :toctree: ../generated

    RoutingMethodType
    GatedActType

**示例：**

.. code-block:: python

    import flashinfer
    from flashinfer.fused_moe import RoutingMethodType, GatedActType

    # 路由方法类型
    routing_method = RoutingMethodType.ROUND_ROBIN
    print(f"路由方法: {routing_method}")

    # 门控激活类型
    act_type = GatedActType.SILU
    print(f"激活类型: {act_type}")

工具函数
--------

辅助 MoE 操作的实用函数。

.. autofunction:: reorder_rows_for_gated_act_gemm

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    hidden_size = 4096
    intermediate_size = 11008

    # 准备输入
    x = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    expert_indices = torch.randint(0, 8, (batch_size,)).int().to("cuda:0")

    # 为门控激活 GEMM 重排行
    reordered_x = flashinfer.reorder_rows_for_gated_act_gemm(x, expert_indices)

CUTLASS 融合 MoE
---------------

使用 CUTLASS 后端的融合 MoE 操作，提供高性能的专家计算。

.. autofunction:: cutlass_fused_moe

**参数说明：**

- ``x``: 输入张量，形状为 ``(num_tokens, hidden_size)``
- ``gating_output``: 门控输出，形状为 ``(num_tokens, num_experts)``
- ``experts``: 专家权重，形状取决于权重布局
- ``num_experts``: 专家数量
- ``top_k``: 每个 token 激活的专家数量
- ``expert_capacity``: 每个专家的容量
- ``hidden_size``: 隐藏层大小
- ``intermediate_size``: 中间层大小
- ``act_type``: 激活函数类型
- ``routing_method``: 路由方法

**示例：**

.. code-block:: python

    import torch
    import flashinfer
    from flashinfer.fused_moe import RoutingMethodType, GatedActType

    batch_size = 8
    num_experts = 8
    top_k = 2
    hidden_size = 4096
    intermediate_size = 11008
    expert_capacity = 64

    # 准备输入
    x = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    
    # 门控输出（路由 logits）
    gating_output = torch.randn(batch_size, num_experts).half().to("cuda:0")
    
    # 专家权重（CUTLASS 布局）
    experts = torch.randn(
        num_experts, 2, intermediate_size, hidden_size,
        dtype=torch.float16, device="cuda:0"
    )

    # 执行融合 MoE
    output = flashinfer.cutlass_fused_moe(
        x, gating_output, experts,
        num_experts=num_experts,
        top_k=top_k,
        expert_capacity=expert_capacity,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        act_type=GatedActType.SILU,
        routing_method=RoutingMethodType.ROUND_ROBIN
    )

    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

TensorRT-LLM 融合 MoE
--------------------

TensorRT-LLM 后端的融合 MoE 操作，支持 FP4 和 FP8 量化。

FP8 块缩放 MoE
~~~~~~~~~~~~~~

.. autofunction:: trtllm_fp8_block_scale_moe

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_experts = 8
    top_k = 2
    hidden_size = 4096
    intermediate_size = 11008

    x = torch.randn(batch_size, hidden_size).half().to("cuda:0")
    gating_output = torch.randn(batch_size, num_experts).half().to("cuda:0")
    
    # FP8 量化专家权重
    experts_fp8 = torch.randn(
        num_experts, 2, intermediate_size, hidden_size,
        dtype=torch.float16, device="cuda:0"
    )
    experts_scale = torch.randn(
        num_experts, 2, intermediate_size // 16
    ).half().to("cuda:0")

    output = flashinfer.trtllm_fp8_block_scale_moe(
        x, gating_output, experts_fp8, experts_scale,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size
    )

FP8 每张量缩放 MoE
~~~~~~~~~~~~~~~~

.. autofunction:: trtllm_fp8_per_tensor_scale_moe

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # FP8 每张量缩放（更简单的缩放方式）
    experts_scale_per_tensor = torch.randn(num_experts, 2).half().to("cuda:0")

    output = flashinfer.trtllm_fp8_per_tensor_scale_moe(
        x, gating_output, experts_fp8, experts_scale_per_tensor,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size
    )

FP4 块缩放 MoE
~~~~~~~~~~~~

.. autofunction:: trtllm_fp4_block_scale_moe

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # FP4 量化专家权重
    experts_fp4 = torch.randint(
        0, 255, (num_experts, 2, intermediate_size, hidden_size),
        dtype=torch.uint8, device="cuda:0"
    )
    experts_scale_fp4 = torch.randn(
        num_experts, 2, intermediate_size // 16
    ).half().to("cuda:0")

    output = flashinfer.trtllm_fp4_block_scale_moe(
        x, gating_output, experts_fp4, experts_scale_fp4,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size
    )

完整 MoE 层示例
--------------

以下是一个完整的 MoE 层示例：

.. code-block:: python

    import torch
    import flashinfer
    from flashinfer.fused_moe import RoutingMethodType, GatedActType

    class MoELayer:
        def __init__(self, num_experts, hidden_size, intermediate_size, top_k=2):
            self.num_experts = num_experts
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.top_k = top_k
            
            # 初始化专家权重（实际应用中从模型加载）
            self.experts = torch.randn(
                num_experts, 2, intermediate_size, hidden_size,
                dtype=torch.float16, device="cuda:0"
            )
            
            # 初始化门控网络（简化）
            self.gating_network = torch.randn(
                hidden_size, num_experts,
                dtype=torch.float16, device="cuda:0"
            )
        
        def forward(self, x):
            """前向传播"""
            batch_size = x.shape[0]
            
            # 计算门控输出
            gating_output = torch.matmul(x, self.gating_network)
            
            # 执行融合 MoE
            output = flashinfer.cutlass_fused_moe(
                x, gating_output, self.experts,
                num_experts=self.num_experts,
                top_k=self.top_k,
                expert_capacity=64,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                act_type=GatedActType.SILU,
                routing_method=RoutingMethodType.ROUND_ROBIN
            )
            
            return output

    # 使用示例
    moe_layer = MoELayer(
        num_experts=8, hidden_size=4096, intermediate_size=11008, top_k=2
    )
    
    x = torch.randn(8, 4096).half().to("cuda:0")
    output = moe_layer.forward(x)
    print(f"输出形状: {output.shape}")  # torch.Size([8, 4096])

性能优化提示
------------

1. **专家容量**：合理设置 expert_capacity 以平衡性能和内存使用
2. **量化**：使用 FP8 或 FP4 量化可以显著减少内存占用
3. **路由方法**：根据模型特性选择合适的路由方法
4. **批量大小**：较大的批量大小可以提高 GPU 利用率

API 参考
--------

.. autosummary::
    :toctree: ../generated

    RoutingMethodType
    GatedActType
    reorder_rows_for_gated_act_gemm
    cutlass_fused_moe
    trtllm_fp4_block_scale_moe
    trtllm_fp8_block_scale_moe
    trtllm_fp8_per_tensor_scale_moe
