"""Stage-2 tests: the full decoder model.

Run:  python tests/test_model.py     (from tiny-moe/)

Checks:
  1. forward shapes + loss sanity
  2. CAUSALITY: logits at position t depend only on tokens <= t
  3. KV-CACHE EQUIVALENCE: incremental decoding == full forward  (the key one)
  4. GQA: KV cache is 4x smaller than MHA would be
  5. RoPE: scaling types produce the expected angles
  6. dense_ffn switch: builds, runs, and param parity with the MoE model
  7. parameter audit: total vs active
  8. speed probe: fwd+bwd at 2k context (for training-time estimates)
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinymoe.model import ModelConfig, TinyMoE                      # noqa: E402
from tinymoe.rope import build_rope_cache                          # noqa: E402

torch.manual_seed(0)


def small_cfg(**kw):
    """Tiny config so tests run in seconds."""
    base = dict(vocab_size=128, d_model=64, n_layers=2, n_heads=4,
                n_kv_heads=2, head_dim=16, max_seq_len=64,
                n_experts=4, top_k=2, ffn_hidden=32)
    base.update(kw)
    return ModelConfig(**base)


def test_forward_shapes():
    cfg = small_cfg()
    m = TinyMoE(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 10))
    out = m(idx, targets=idx)
    assert out.logits.shape == (2, 10, cfg.vocab_size)
    assert out.loss.ndim == 0 and out.loss.item() > 0
    assert out.aux_loss.ndim == 0
    assert len(out.expert_counts) == cfg.n_layers
    print("  [ok] forward shapes + loss")


def test_causality():
    """Changing a future token must not change past logits."""
    cfg = small_cfg()
    m = TinyMoE(cfg).eval()
    a = torch.randint(0, cfg.vocab_size, (1, 20))
    b = a.clone()
    b[0, 15:] = (b[0, 15:] + 1) % cfg.vocab_size     # different suffix
    with torch.no_grad():
        la = m(a).logits
        lb = m(b).logits
    err = (la[0, :15] - lb[0, :15]).abs().max().item()
    assert err < 1e-5, f"leak across positions: {err}"
    assert (la[0, 15:] - lb[0, 15:]).abs().max().item() > 1e-3
    print(f"  [ok] causality (past logits identical, max err {err:.1e})")


def test_kv_cache_equivalence():
    """Incremental decode must produce exactly the full-forward logits."""
    cfg = small_cfg()
    m = TinyMoE(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 12))

    with torch.no_grad():
        full = m(idx).logits                      # (1, 12, vocab)

        # step 1: prefill the first 5 tokens
        out = m(idx[:, :5])
        cache = out.cache
        # step 2: feed the rest one token at a time
        step_logits = [out.logits[:, -1:]]
        for t in range(5, 12):
            out = m(idx[:, t:t + 1], cache=cache)
            cache = out.cache
            step_logits.append(out.logits)
        inc = torch.cat(step_logits, dim=1)       # (1, 8, vocab): positions 4..11

    # incremental covers positions 4..11 (last prefill pos + 7 decode steps)
    err = (full[:, 4:] - inc).abs().max().item()
    assert err < 1e-4, f"cache mismatch: {err}"
    assert cache[0][0].shape[2] == 12             # cache grew to 12 tokens
    print(f"  [ok] incremental KV cache == full forward (max err {err:.1e})")


def test_gqa_cache_size():
    """Real config: 2 KV heads vs 8 = exactly 4x cache saving.

    Arithmetic (fp32, 8192 tokens, batch 1):
      cache bytes = 2 (K and V) * n_layers * n_kv_heads * head_dim
                    * 4 bytes * 8192
    GQA (2 kv heads): 2 * 6 * 2 * 64 * 4 * 8192 =  50,331,648 = 50 MB
    MHA (8 kv heads): 2 * 6 * 8 * 64 * 4 * 8192 = 201,326,592 = 201 MB
    -> exactly 4x.

    (An earlier version of this test compared 2 vs 4 kv heads in a 4-head
    tiny config -- arithmetically right, but the 'GQA vs MHA' label implied
    the real 2-vs-8 ratio. This version tests the real config.)
    """
    cfg_gqa = ModelConfig()                 # n_heads=8, n_kv_heads=2
    cfg_mha = ModelConfig(n_kv_heads=8)     # pure MHA variant
    mg, mm = TinyMoE(cfg_gqa), TinyMoE(cfg_mha)
    b_g, b_m = mg.kv_cache_bytes(8192), mm.kv_cache_bytes(8192)
    expected_gqa = 2 * 6 * 2 * 64 * 4 * 8192
    assert b_g == expected_gqa, (b_g, expected_gqa)
    assert b_m == 4 * b_g, (b_g, b_m)
    kv_params_g = sum(p.numel() for n, p in mg.named_parameters()
                      if ".wk." in n or ".wv." in n)
    kv_params_m = sum(p.numel() for n, p in mm.named_parameters()
                      if ".wk." in n or ".wv." in n)
    assert kv_params_m == 4 * kv_params_g
    print(f"  [ok] GQA cache @8k fp32: {b_g/1e6:.0f} MB (2 kv heads) vs "
          f"{b_m/1e6:.0f} MB MHA (8 kv heads) = {b_m // b_g}x saving")


def test_rope_scaling():
    # "none": angle(pos, i) == pos * base^(-2i/d)
    cos, sin = build_rope_cache(8, 16, base=10000.0, scaling_type="none")
    i = torch.arange(8).float()
    inv_freq = 10000.0 ** (-i / 8)
    expected = (torch.arange(8).float()[:, None] * inv_freq[None, :]).cos()
    assert torch.allclose(cos[:, :8], expected, atol=1e-6)

    # "linear" with s=4 at pos 8 == "none" at pos 2
    cos_l, _ = build_rope_cache(16, 16, scaling_type="linear",
                                scaling_factor=4.0)
    assert torch.allclose(cos_l[8], cos[2], atol=1e-6)

    # "ntk" changes the frequencies (and only them)
    cos_n, _ = build_rope_cache(8, 16, scaling_type="ntk", scaling_factor=4.0)
    assert not torch.allclose(cos_n[1], cos[1], atol=1e-3)
    print("  [ok] rope: none/linear/ntk angles behave as specified")


def test_dense_parity():
    """MoE model vs dense baseline at same TOTAL parameter count."""
    cfg_moe = small_cfg(n_experts=4, ffn_hidden=32)         # 4*32 = 128
    cfg_dense = small_cfg(dense_ffn=True, ffn_hidden=128)
    mm, md = TinyMoE(cfg_moe), TinyMoE(cfg_dense)
    p_moe, p_dense = mm.num_params(), md.num_params()
    # embedding dominates in the tiny test config; FFN parts should match
    ffn_moe = sum(p.numel() for n, p in mm.named_parameters()
                  if ".experts." in n)
    ffn_dense = sum(p.numel() for n, p in md.named_parameters()
                    if ".ffn.ffn." in n)
    assert ffn_moe == ffn_dense, (ffn_moe, ffn_dense)
    idx = torch.randint(0, cfg_moe.vocab_size, (2, 8))
    assert md(idx).logits.shape == mm(idx).logits.shape
    print(f"  [ok] dense_ffn switch: FFN params match exactly "
          f"({ffn_moe/1e6:.2f}M each); totals {p_moe/1e6:.1f}M vs {p_dense/1e6:.1f}M")


def test_param_audit():
    cfg = ModelConfig()   # the real 34M config
    m = TinyMoE(cfg)
    total = m.num_params()
    active = m.num_params(active=True)
    print(f"  [ok] real config: total {total/1e6:.1f}M, active/token "
          f"{active/1e6:.1f}M ({100*active/total:.0f}%), "
          f"kv cache @8k fp32 = {m.kv_cache_bytes(8192)/1e6:.0f} MB, "
          f"@8k int8 = {m.kv_cache_bytes(8192, 1)/1e6:.0f} MB")
    assert 30e6 < total < 40e6


def bench():
    """fwd+bwd at 2k context, batch 1 -- anchors the training-time estimate."""
    print("\nProbe -- full model fwd+bwd @ 2k (this machine, 3 vCPU):")
    cfg = ModelConfig()
    m = TinyMoE(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 2048))
    out = m(idx, targets=idx)
    (out.loss + 0.01 * out.aux_loss).backward()
    m.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(3):
        out = m(idx, targets=idx)
        (out.loss + cfg.aux_loss_coef * out.aux_loss).backward()
        m.zero_grad(set_to_none=True)
    dt = (time.perf_counter() - t0) / 3
    print(f"  batch=1 x 2048 tokens: {dt*1000:.0f} ms/step -> "
          f"{2048/dt:.0f} tokens/sec")


if __name__ == "__main__":
    print("Running Stage-2 model tests:")
    test_forward_shapes()
    test_causality()
    test_kv_cache_equivalence()
    test_gqa_cache_size()
    test_rope_scaling()
    test_dense_parity()
    test_param_audit()
    bench()
    print("\nAll tests passed.")
