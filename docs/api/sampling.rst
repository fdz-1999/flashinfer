.. _apisampling:

flashinfer.sampling
===================

FlashInfer 提供了高效的 LLM 采样内核，支持多种采样策略，包括 Top-K、Top-P、Min-P 等。这些内核经过优化，无需排序即可实现高性能采样。

主要特性：
- 无需排序的高效采样算法
- 支持多种采样策略（Top-K、Top-P、Min-P）
- 支持批量采样和索引映射
- 支持温度缩放和概率归一化
- 支持推测性解码的链式采样

.. currentmodule:: flashinfer.sampling

Softmax
-------

计算 softmax 概率分布，支持温度缩放。

.. autofunction:: softmax

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 4
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")

    # 使用固定温度
    probs = flashinfer.softmax(logits, temperature=1.0)

    # 使用每个请求不同的温度
    temperatures = torch.tensor([0.8, 1.0, 1.2, 0.9]).to("cuda:0")
    probs = flashinfer.softmax(logits, temperature=temperatures)

基础采样
--------

从 logits 或概率分布中采样。

.. autofunction:: sampling_from_logits

.. autofunction:: sampling_from_probs

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")

    # 从 logits 采样（内部会先计算 softmax）
    samples = flashinfer.sampling_from_logits(logits, deterministic=False)

    # 从概率分布采样
    probs = flashinfer.softmax(logits, temperature=1.0)
    samples = flashinfer.sampling_from_probs(probs, deterministic=False)

    # 使用索引映射（多个输出共享同一个概率分布）
    indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32, device="cuda:0")
    unique_logits = torch.randn(4, vocab_size).half().to("cuda:0")
    samples = flashinfer.sampling_from_logits(unique_logits, indices=indices)

Top-P 采样
----------

Top-P（核采样）从累积概率达到阈值 p 的最小 token 集合中采样。

.. autofunction:: top_p_sampling_from_probs

.. autofunction:: top_p_renorm_probs

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")
    probs = flashinfer.softmax(logits, temperature=1.0)

    # Top-P 采样（p=0.9）
    samples = flashinfer.top_p_sampling_from_probs(probs, top_p=0.9)

    # 先归一化概率，再采样
    renorm_probs = flashinfer.top_p_renorm_probs(probs, top_p=0.9)
    samples = flashinfer.sampling_from_probs(renorm_probs)

Top-K 采样
----------

Top-K 采样从概率最高的 k 个 token 中采样。

.. autofunction:: top_k_sampling_from_probs

.. autofunction:: top_k_renorm_probs

.. autofunction:: top_k_mask_logits

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")
    probs = flashinfer.softmax(logits, temperature=1.0)

    # Top-K 采样（k=50）
    samples = flashinfer.top_k_sampling_from_probs(probs, top_k=50)

    # 先归一化概率，再采样
    renorm_probs = flashinfer.top_k_renorm_probs(probs, top_k=50)
    samples = flashinfer.sampling_from_probs(renorm_probs)

    # 直接掩码 logits（用于自定义采样流程）
    masked_logits = flashinfer.top_k_mask_logits(logits, top_k=50)

Min-P 采样
----------

Min-P 采样保留概率至少为最大概率的 p 倍的所有 token。

.. autofunction:: min_p_sampling_from_probs

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")
    probs = flashinfer.softmax(logits, temperature=1.0)

    # Min-P 采样（min_p=0.05）
    samples = flashinfer.min_p_sampling_from_probs(probs, min_p=0.05)

组合采样
--------

同时应用 Top-K 和 Top-P 的采样策略。

.. autofunction:: top_k_top_p_sampling_from_logits

.. autofunction:: top_k_top_p_sampling_from_probs

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    vocab_size = 50000
    logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")

    # 从 logits 进行 Top-K + Top-P 采样
    samples = flashinfer.top_k_top_p_sampling_from_logits(
        logits, top_k=50, top_p=0.9, temperature=1.0
    )

    # 从概率分布进行 Top-K + Top-P 采样
    probs = flashinfer.softmax(logits, temperature=1.0)
    samples = flashinfer.top_k_top_p_sampling_from_probs(
        probs, top_k=50, top_p=0.9
    )

推测性解码
----------

支持链式推测性采样，用于加速 LLM 推理。

.. autofunction:: chain_speculative_sampling

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    vocab_size = 50000
    draft_logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")
    target_logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")

    # 链式推测性采样
    accepted, samples = flashinfer.chain_speculative_sampling(
        draft_logits, target_logits, temperature=1.0
    )

完整使用示例
------------

以下是一个完整的 LLM 生成循环示例：

.. code-block:: python

    import torch
    import flashinfer

    def generate_token(logits, temperature=1.0, top_k=50, top_p=0.9):
        """生成单个 token"""
        # 方法1：使用组合采样（推荐）
        samples = flashinfer.top_k_top_p_sampling_from_logits(
            logits, top_k=top_k, top_p=top_p, temperature=temperature
        )
        return samples

        # 方法2：分步采样
        # probs = flashinfer.softmax(logits, temperature=temperature)
        # renorm_probs = flashinfer.top_k_renorm_probs(probs, top_k=top_k)
        # renorm_probs = flashinfer.top_p_renorm_probs(renorm_probs, top_p=top_p)
        # samples = flashinfer.sampling_from_probs(renorm_probs)
        # return samples

    # 模拟生成过程
    batch_size = 4
    vocab_size = 50000

    for step in range(10):
        # 获取模型输出的 logits
        logits = torch.randn(batch_size, vocab_size).half().to("cuda:0")
        
        # 采样生成 token
        tokens = generate_token(logits, temperature=0.8, top_k=50, top_p=0.9)
        print(f"Step {step}: {tokens}")

API 参考
--------

.. autosummary::
    :toctree: ../generated

    softmax
    sampling_from_probs
    sampling_from_logits
    top_p_sampling_from_probs
    top_k_sampling_from_probs
    min_p_sampling_from_probs
    top_k_top_p_sampling_from_logits
    top_k_top_p_sampling_from_probs
    top_p_renorm_probs
    top_k_renorm_probs
    top_k_mask_logits
    chain_speculative_sampling
