.. _apiquantization:

flashinfer.quantization
=======================

FlashInfer 提供了量化相关的内核，用于压缩和打包数据。

主要特性：
- 位打包操作
- 分段位打包
- 支持不同的位序

.. currentmodule:: flashinfer.quantization

位打包
------

将数据打包为位格式以节省内存。

.. autofunction:: packbits

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # 准备二进制数据
    x = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0], dtype=torch.uint8, device="cuda:0")

    # 打包为位（大端序）
    packed = flashinfer.packbits(x, bitorder="big")
    print(f"打包后形状: {packed.shape}")

    # 打包为位（小端序）
    packed_little = flashinfer.packbits(x, bitorder="little")

分段位打包
----------

对分段数据进行位打包。

.. autofunction:: segment_packbits

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    # 准备分段数据
    x = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.uint8, device="cuda:0")
    segment_sizes = torch.tensor([5, 5], dtype=torch.int32, device="cuda:0")

    # 分段打包
    packed = flashinfer.segment_packbits(x, segment_sizes, bitorder="big")

完整量化示例
-----------

以下是一个使用量化操作的完整示例：

.. code-block:: python

    import torch
    import flashinfer

    def quantize_and_pack(data, bits_per_element=4):
        """量化和打包数据"""
        # 量化到指定位数（简化示例）
        quantized = (data * (2 ** bits_per_element - 1)).int()
        
        # 打包为位
        if bits_per_element == 1:
            packed = flashinfer.packbits(quantized.byte(), bitorder="big")
        else:
            # 对于多位量化，需要更复杂的打包逻辑
            packed = quantized
        
        return packed, quantized

    # 使用示例
    data = torch.randn(100).half().to("cuda:0")
    packed, quantized = quantize_and_pack(data, bits_per_element=4)

API 参考
--------

.. autosummary::
    :toctree: ../generated

    packbits
    segment_packbits
