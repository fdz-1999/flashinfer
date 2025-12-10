.. _apipage:

flashinfer.page
===============

FlashInfer 提供了操作分页 KV 缓存的内核。分页 KV 缓存是 vLLM 引入的内存高效 KV 缓存管理方式，允许动态分配和管理 KV 缓存页面。

主要特性：
- 高效的分页 KV 缓存追加操作
- 支持标准注意力和 MLA 注意力
- 批量索引和位置计算
- 内存高效的缓存管理

.. currentmodule:: flashinfer.page

追加到分页 KV 缓存
-----------------

将新的 K/V 张量追加到现有的分页 KV 缓存中。

.. autofunction:: append_paged_kv_cache

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_kv_heads = 32
    head_dim = 128
    page_size = 16
    max_num_pages = 256

    # 准备分页 KV 缓存
    kv_cache = torch.randn(
        max_num_pages, 2, page_size, num_kv_heads, head_dim,
        dtype=torch.float16, device="cuda:0"
    )

    # 准备新的 K/V 值（要追加的 token）
    new_k = torch.randn(batch_size, num_kv_heads, head_dim).half().to("cuda:0")
    new_v = torch.randn(batch_size, num_kv_heads, head_dim).half().to("cuda:0")

    # 准备页面表
    page_table = torch.randint(0, max_num_pages, (batch_size, 1)).int().to("cuda:0")
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

    # 追加到缓存
    flashinfer.append_paged_kv_cache(
        new_k, new_v, kv_cache, page_table, last_page_len, page_size
    )

MLA 分页 KV 缓存
---------------

支持 MLA（Multi-head Latent Attention）的分页 KV 缓存操作。

.. autofunction:: append_paged_mla_kv_cache

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    num_kv_heads = 32
    head_dim = 128
    page_size = 16
    max_num_pages = 256

    # MLA KV 缓存（包含压缩的 KV）
    mla_kv_cache = torch.randn(
        max_num_pages, page_size, num_kv_heads, head_dim * 2,
        dtype=torch.float16, device="cuda:0"
    )

    new_k = torch.randn(batch_size, num_kv_heads, head_dim).half().to("cuda:0")
    new_v = torch.randn(batch_size, num_kv_heads, head_dim).half().to("cuda:0")

    page_table = torch.randint(0, max_num_pages, (batch_size, 1)).int().to("cuda:0")
    last_page_len = torch.ones(batch_size, dtype=torch.int32, device="cuda:0")

    # 追加到 MLA 缓存
    flashinfer.append_paged_mla_kv_cache(
        new_k, new_v, mla_kv_cache, page_table, last_page_len, page_size
    )

批量索引和位置
-------------

获取批量请求的索引和位置信息。

.. autofunction:: get_batch_indices_positions

.. autofunction:: get_seq_lens

**示例：**

.. code-block:: python

    import torch
    import flashinfer

    batch_size = 8
    max_num_pages = 256
    page_size = 16

    # 准备页面表
    page_table = torch.randint(0, max_num_pages, (batch_size, 20)).int().to("cuda:0")
    last_page_len = torch.randint(1, page_size + 1, (batch_size,)).int().to("cuda:0")

    # 获取序列长度
    seq_lens = flashinfer.get_seq_lens(page_table, last_page_len, page_size)
    print(f"序列长度: {seq_lens}")

    # 获取批量索引和位置
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        page_table, last_page_len, page_size
    )

完整 KV 缓存管理示例
--------------------

以下是一个完整的分页 KV 缓存管理示例：

.. code-block:: python

    import torch
    import flashinfer

    class PagedKVCacheManager:
        def __init__(self, max_num_pages, page_size, num_kv_heads, head_dim):
            self.max_num_pages = max_num_pages
            self.page_size = page_size
            self.num_kv_heads = num_kv_heads
            self.head_dim = head_dim
            
            # 分配 KV 缓存
            self.kv_cache = torch.zeros(
                max_num_pages, 2, page_size, num_kv_heads, head_dim,
                dtype=torch.float16, device="cuda:0"
            )
            
            # 页面分配表（简化版本，实际应用中需要更复杂的管理）
            self.page_allocations = {}
        
        def append(self, request_id, k, v):
            """追加新的 K/V 到指定请求的缓存"""
            if request_id not in self.page_allocations:
                # 分配新页面（简化版本）
                page_idx = len(self.page_allocations)
                self.page_allocations[request_id] = {
                    'pages': [page_idx],
                    'last_page_len': 0
                }
            
            alloc = self.page_allocations[request_id]
            page_table = torch.tensor(alloc['pages'], dtype=torch.int32, device="cuda:0").unsqueeze(0)
            last_page_len = torch.tensor([alloc['last_page_len']], dtype=torch.int32, device="cuda:0")
            
            # 追加到缓存
            flashinfer.append_paged_kv_cache(
                k.unsqueeze(0), v.unsqueeze(0),
                self.kv_cache, page_table, last_page_len, self.page_size
            )
            
            # 更新最后页面长度
            alloc['last_page_len'] += 1
            if alloc['last_page_len'] >= self.page_size:
                # 分配新页面
                new_page = len(self.page_allocations)
                alloc['pages'].append(new_page)
                alloc['last_page_len'] = 0
        
        def get_page_table(self, request_id):
            """获取请求的页面表"""
            if request_id not in self.page_allocations:
                return None
            alloc = self.page_allocations[request_id]
            return torch.tensor(alloc['pages'], dtype=torch.int32, device="cuda:0")
        
        def get_seq_len(self, request_id):
            """获取请求的序列长度"""
            if request_id not in self.page_allocations:
                return 0
            alloc = self.page_allocations[request_id]
            page_table = self.get_page_table(request_id).unsqueeze(0)
            last_page_len = torch.tensor([alloc['last_page_len']], dtype=torch.int32, device="cuda:0")
            seq_lens = flashinfer.get_seq_lens(page_table, last_page_len, self.page_size)
            return seq_lens[0].item()

    # 使用示例
    cache_manager = PagedKVCacheManager(
        max_num_pages=256, page_size=16, num_kv_heads=32, head_dim=128
    )
    
    # 为请求追加 KV
    request_id = 0
    k = torch.randn(32, 128).half().to("cuda:0")
    v = torch.randn(32, 128).half().to("cuda:0")
    cache_manager.append(request_id, k, v)
    
    # 获取序列长度
    seq_len = cache_manager.get_seq_len(request_id)
    print(f"序列长度: {seq_len}")

API 参考
--------

.. autosummary::
  :toctree: ../generated

  append_paged_kv_cache
  append_paged_mla_kv_cache
  get_batch_indices_positions
  get_seq_lens
