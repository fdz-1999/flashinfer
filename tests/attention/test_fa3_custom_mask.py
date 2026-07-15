"""FA3 Single-Prefill Custom Mask Precision Tests

Cross-check FA3 custom (packed-bitmask) mask against the FA2 reference.
Only runs on SM90+.

Mask rule: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)

Test coverage:
  - Multiple shapes × block sizes × offsets
  - Both fp16 and bf16 dtypes
"""

import math

import torch
import pytest
import flashinfer
from flashinfer import single_prefill_with_kv_cache


def _has_fa3(device=None):
    """Check whether FA3 backend is available (requires SM90+)."""
    if device is None:
        device = torch.device("cuda:0")
    return flashinfer.utils.is_sm90a_supported(device)


@pytest.mark.parametrize(
    "qo_len,kv_len,num_heads,num_kv_heads,head_dim,dllm_block_size,q_offset",
    [
        (64, 128, 32, 8, 128, 16, 0),
        (64, 128, 32, 8, 128, 32, 0),
        (64, 128, 32, 8, 128, 64, 0),
        (64, 192, 32, 8, 128, 16, 64),
        (128, 2048, 32, 8, 128, 32, 0),
        (33, 97, 32, 4, 128, 16, 0),
        (64, 128, 32, 8, 64, 32, 0),
        (32, 32, 32, 1, 128, 16, 0),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fa3_custom_mask_vs_fa2(
    qo_len, kv_len, num_heads, num_kv_heads, head_dim, dllm_block_size, q_offset, dtype
):
    """Cross-check FA3 custom_mask against FA2 custom_mask reference."""
    device = torch.device("cuda:0")
    if not _has_fa3(device):
        pytest.skip("FA3 requires SM90+")

    tol = 1e-2 if dtype == torch.float16 else 2e-2
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Block-extend mask (the DLLM rule).
    q_pos = torch.arange(qo_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    mask_2d = (
        (q_pos.unsqueeze(1) // dllm_block_size)
        >= (k_pos.unsqueeze(0) // dllm_block_size)
    ).to(torch.uint8)

    ref_fa2 = single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2"
    )
    ref_fa3 = single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3"
    )

    max_diff = (ref_fa3 - ref_fa2).abs().max().item()
    assert max_diff < tol, (
        f"FA3 vs FA2 mismatch: max_diff={max_diff:.6f} > tol={tol:.0e} "
        f"(qo={qo_len}, kv={kv_len}, B={dllm_block_size}, off={q_offset}, "
        f"h={num_heads}/{num_kv_heads}, d={head_dim}, {dtype})"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fa3_custom_mask_causal(dtype):
    """Check FA3 custom_mask with a causal pattern against FA2."""
    device = torch.device("cuda:0")
    if not _has_fa3(device):
        pytest.skip("FA3 requires SM90+")

    tol = 1e-2 if dtype == torch.float16 else 2e-2
    head_dim = 128
    qo_len, kv_len = 128, 256
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(qo_len, 32, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, 8, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, 8, head_dim, dtype=dtype, device=device)

    # Causal tile mask (j <= i + offset, where offset = kv_len - qo_len)
    offset = kv_len - qo_len
    i = torch.arange(qo_len, device=device).unsqueeze(1)  # [qo_len, 1]
    j = torch.arange(kv_len, device=device).unsqueeze(0)  # [1, kv_len]
    mask_2d = (j <= i + offset).to(torch.uint8)

    ref_fa2 = single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2"
    )
    ref_fa3 = single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa3"
    )

    max_diff = (ref_fa3 - ref_fa2).abs().max().item()
    assert max_diff < tol, f"FA3 custom_mask causal vs FA2 mismatch: {max_diff:.6f}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fa3_custom_mask_full(dtype):
    """Check FA3 custom_mask with all-ones (no masking) vs FA3 causal."""
    device = torch.device("cuda:0")
    if not _has_fa3(device):
        pytest.skip("FA3 requires SM90+")

    tol = 1e-2 if dtype == torch.float16 else 2e-2
    head_dim = 128
    qo_len, kv_len = 64, 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(qo_len, 32, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, 8, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, 8, head_dim, dtype=dtype, device=device)

    # All-ones mask: nothing masked out (matches non-causal FA3).
    mask_all = torch.ones(qo_len, kv_len, dtype=torch.uint8, device=device)

    # FA3 with custom mask (all ones => everything visible).
    out_masked = single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_all, sm_scale=sm_scale, backend="fa3"
    )
    # FA3 without mask (non-causal => everything visible).
    out_nomask = single_prefill_with_kv_cache(
        q, k, v, causal=False, sm_scale=sm_scale, backend="fa3"
    )

    max_diff = (out_masked - out_nomask).abs().max().item()
    assert max_diff < tol, (
        f"FA3 all-ones custom_mask vs no-mask mismatch: {max_diff:.6f}"
    )