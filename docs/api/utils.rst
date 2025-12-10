.. _apiutils:

flashinfer.utils
================

FlashInfer 提供了一系列实用工具函数，用于辅助 LLM 推理任务。这些工具函数涵盖了数据类型检查、设备信息查询、张量操作等功能。

.. currentmodule:: flashinfer.utils

数据类型和枚举
--------------

.. autoclass:: PosEncodingMode
    :members:

.. autoclass:: MaskMode
    :members:

.. autoclass:: TensorLayout
    :members:

**示例：**

.. code-block:: python

    import flashinfer
    from flashinfer.utils import PosEncodingMode, MaskMode, TensorLayout

    # 使用位置编码模式
    mode = PosEncodingMode.ROPE_LLAMA
    print(f"位置编码模式: {mode}")

    # 使用掩码模式
    mask = MaskMode.CAUSAL
    print(f"掩码模式: {mask}")

    # 使用张量布局
    layout = TensorLayout.NHD
    print(f"张量布局: {layout}")

设备信息查询
------------

查询 GPU 设备的相关信息。

.. autofunction:: get_compute_capability

.. autofunction:: get_device_sm_count

.. autofunction:: get_gpu_memory_bandwidth

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    device = torch.device("cuda:0")
    
    # 获取计算能力
    major, minor = flashinfer.get_compute_capability(device)
    print(f"计算能力: {major}.{minor}")
    
    # 获取 SM 数量
    sm_count = flashinfer.get_device_sm_count(device)
    print(f"SM 数量: {sm_count}")
    
    # 获取内存带宽（GB/s）
    bandwidth = flashinfer.get_gpu_memory_bandwidth(device)
    print(f"内存带宽: {bandwidth} GB/s")

数据类型检查
------------

检查张量的数据类型和格式。

.. autofunction:: is_float8

.. autofunction:: canonicalize_torch_dtype

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    x = torch.randn(10, 10).half().to("cuda:0")
    
    # 检查是否为 FP8
    is_fp8 = flashinfer.is_float8(x)
    print(f"是否为 FP8: {is_fp8}")
    
    # 规范化数据类型
    dtype = flashinfer.canonicalize_torch_dtype("float16")
    print(f"规范化后的类型: {dtype}")

张量操作
--------

辅助张量操作的实用函数。

.. autofunction:: next_positive_power_of_2

.. autofunction:: get_indptr

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # 获取下一个 2 的幂
    x = 100
    next_pow2 = flashinfer.next_positive_power_of_2(x)
    print(f"{x} 的下一个 2 的幂: {next_pow2}")  # 128
    
    # 从序列长度获取 indptr
    seq_lens = torch.tensor([10, 20, 30, 15])
    indptr = flashinfer.get_indptr(seq_lens)
    print(f"Indptr: {indptr}")  # [0, 10, 30, 60, 75]

后端检测
--------

检测和选择可用的计算后端。

.. autofunction:: determine_attention_backend

.. autofunction:: determine_gemm_backend

.. autofunction:: determine_mla_backend

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    device = torch.device("cuda:0")
    
    # 确定注意力后端
    attn_backend = flashinfer.determine_attention_backend(device)
    print(f"注意力后端: {attn_backend}")
    
    # 确定 GEMM 后端
    gemm_backend = flashinfer.determine_gemm_backend(device)
    print(f"GEMM 后端: {gemm_backend}")
    
    # 确定 MLA 后端
    mla_backend = flashinfer.determine_mla_backend(device)
    print(f"MLA 后端: {mla_backend}")

架构支持检查
------------

检查 GPU 架构是否支持特定功能。

.. autofunction:: is_sm90a_supported

.. autofunction:: is_sm100a_supported

.. autofunction:: is_sm100f_supported

.. autofunction:: is_sm110a_supported

.. autofunction:: is_sm120a_supported

.. autofunction:: is_sm121a_supported

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    device = torch.device("cuda:0")
    
    # 检查架构支持
    print(f"SM90A 支持: {flashinfer.is_sm90a_supported(device)}")
    print(f"SM100A 支持: {flashinfer.is_sm100a_supported(device)}")
    print(f"SM100F 支持: {flashinfer.is_sm100f_supported(device)}")
    print(f"SM110A 支持: {flashinfer.is_sm110a_supported(device)}")
    print(f"SM120A 支持: {flashinfer.is_sm120a_supported(device)}")
    print(f"SM121A 支持: {flashinfer.is_sm121a_supported(device)}")

数学工具函数
------------

.. autofunction:: ceil_div

.. autofunction:: round_up

**示例：**

.. code-block:: python

    import flashinfer

    # 向上取整除法
    result = flashinfer.ceil_div(100, 16)
    print(f"100 / 16 (向上取整): {result}")  # 7
    
    # 向上舍入到指定倍数
    result = flashinfer.round_up(100, 16)
    print(f"100 向上舍入到 16 的倍数: {result}")  # 112

错误处理
--------

.. autoclass:: GPUArchitectureError
    :members:

.. autoclass:: LibraryError
    :members:

.. autoclass:: BackendSupportedError
    :members:

**示例：**

.. code-block:: python

    import flashinfer
    from flashinfer.utils import GPUArchitectureError

    try:
        # 某些操作可能需要特定的 GPU 架构
        result = some_operation()
    except GPUArchitectureError as e:
        print(f"GPU 架构不支持: {e}")

完整工具函数示例
--------------

以下是一个使用多个工具函数的完整示例：

.. code-block:: python

    import torch
    import flashinfer

    def setup_inference_environment():
        """设置推理环境"""
        device = torch.device("cuda:0")
        
        # 检查设备信息
        major, minor = flashinfer.get_compute_capability(device)
        sm_count = flashinfer.get_device_sm_count(device)
        
        print(f"GPU 信息:")
        print(f"  计算能力: {major}.{minor}")
        print(f"  SM 数量: {sm_count}")
        
        # 确定后端
        attn_backend = flashinfer.determine_attention_backend(device)
        gemm_backend = flashinfer.determine_gemm_backend(device)
        
        print(f"后端选择:")
        print(f"  注意力后端: {attn_backend}")
        print(f"  GEMM 后端: {gemm_backend}")
        
        # 检查架构支持
        if flashinfer.is_sm100a_supported(device):
            print("支持 SM100A 特性")
        if flashinfer.is_sm120a_supported(device):
            print("支持 SM120A 特性")
        
        return device, attn_backend, gemm_backend

    # 使用示例
    device, attn_backend, gemm_backend = setup_inference_environment()

API 参考
--------

.. autosummary::
    :toctree: ../generated

    next_positive_power_of_2
    get_compute_capability
    get_device_sm_count
    get_gpu_memory_bandwidth
    is_float8
    canonicalize_torch_dtype
    get_indptr
    determine_attention_backend
    determine_gemm_backend
    determine_mla_backend
    ceil_div
    round_up
    is_sm90a_supported
    is_sm100a_supported
    is_sm100f_supported
    is_sm110a_supported
    is_sm120a_supported
    is_sm121a_supported
