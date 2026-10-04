"""Correctness tests + a small demo for the Stage-1 MoE layer.

Run:  python tests/test_moe.py          (from the tiny-moe/ directory)

Tests:
  1. shape round-trip
  2. scatter dispatch == naive "sum over top-k experts" reference  (the big one)
  3. aux loss ~ 1.0 at init (near-uniform router = balanced)
  4. aux loss >> 1.0 for a deliberately collapsed router
  5. gradients reach the router AND every expert that got tokens
  6. gate weights per token sum to 1 (checked via the reference path)

Then a demo: expert utilization histogram + tokens/sec on this machine.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinymoe.moe import MoEConfig, MoELayer           # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------------------
# Naive reference implementation (slow but obviously correct)
# --------------------------------------------------------------------------
@torch.no_grad()
def reference_forward(moe, x):
    """out[t] = sum_j gate[t,j] * expert_{idx[t,j]}(x[t]), straight from the
    definition. Used to validate the fast sort/dispatch path."""
    B, T, d = x.shape
    flat = x.reshape(B * T, d)
    logits = moe.router.gate(flat)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_idx = probs.topk(moe.cfg.top_k, dim=-1)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

    out = torch.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(moe.cfg.top_k):
            e = int(top_idx[t, j])
            out[t] += top_probs[t, j] * moe.experts[e](flat[t:t + 1])[0]
    return out.view(B, T, d)


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_shapes():
    cfg = MoEConfig(d_model=64, ffn_hidden=48, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    x = torch.randn(3, 10, 64)
    out = moe(x)
    assert out.output.shape == x.shape, out.output.shape
    assert out.aux_loss.ndim == 0
    assert out.expert_counts.shape == (8,)
    assert int(out.expert_counts.sum()) == 3 * 10 * 2
    print("  [ok] shapes")


def test_dispatch_matches_reference():
    cfg = MoEConfig(d_model=32, ffn_hidden=24, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    x = torch.randn(2, 7, 32)
    fast = moe(x).output
    ref = reference_forward(moe, x)
    max_err = (fast - ref).abs().max().item()
    assert max_err < 1e-5, f"dispatch mismatch: {max_err}"
    print(f"  [ok] sort/dispatch == naive reference (max err {max_err:.2e})")


def test_aux_loss_balanced_at_init():
    cfg = MoEConfig(d_model=64, ffn_hidden=48, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    x = torch.randn(4, 32, 64)          # 128 tokens, random
    aux = moe(x).aux_loss.item()
    assert 0.8 < aux < 1.5, f"expected ~1.0 at init, got {aux}"
    print(f"  [ok] aux loss at init = {aux:.3f} (uniform baseline is 1.0)")


def test_aux_loss_detects_collapse():
    cfg = MoEConfig(d_model=64, ffn_hidden=48, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    # Force the worst possible collapse: every token routes to experts 0/1.
    with torch.no_grad():
        moe.router.gate.weight.zero_()
        moe.router.gate.weight[0].fill_(1.0)
        moe.router.gate.weight[1].fill_(1.0)
    x = torch.ones(4, 32, 64)              # constant input -> deterministic
    out = moe(x)
    aux = out.aux_loss.item()
    # Interesting property of top-k aux loss: its ceiling is E/k = 8/2 = 4,
    # not E. Even a fully collapsed router must pick k experts per token, so
    # usage can never concentrate harder than on k experts. Balanced is 1.0,
    # so the useful signal range is [1.0, E/k].
    assert 3.5 < aux <= 4.01, f"collapse not detected: aux={aux}"
    used = set(int(i) for i in out.expert_counts.nonzero().flatten())
    assert used == {0, 1}, f"expected only experts 0,1, got {used}"
    print(f"  [ok] aux loss under collapse = {aux:.3f} "
          f"(ceiling is E/k = {cfg.n_experts // cfg.top_k})")


def test_gradients():
    cfg = MoEConfig(d_model=64, ffn_hidden=48, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)
    out = moe(x)
    loss = out.output.sum() + out.aux_loss
    loss.backward()

    assert moe.router.gate.weight.grad is not None
    assert x.grad is not None
    for e in range(cfg.n_experts):
        used = int(out.expert_counts[e]) > 0
        grad = moe.experts[e].w1.weight.grad
        if used:
            assert grad is not None and grad.abs().sum() > 0, f"expert {e}"
        # unused experts legitimately get no gradient this step
    print("  [ok] gradients reach router, experts, and inputs")


# --------------------------------------------------------------------------
# Demo: utilization + throughput
# --------------------------------------------------------------------------
def demo_utilization():
    print("\nDemo -- expert utilization (healthy router):")
    cfg = MoEConfig(d_model=128, ffn_hidden=96, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    x = torch.randn(4, 64, 128)          # 256 tokens -> 512 routing slots
    out = moe(x)
    share = out.expert_counts.float() / out.expert_counts.sum()
    for e in range(cfg.n_experts):
        bar = "#" * int(share[e] * 80)
        print(f"  expert {e}: {int(out.expert_counts[e]):4d} slots "
              f"({share[e] * 100:4.1f}%) {bar}")
    print(f"  aux loss = {out.aux_loss:.3f}  (1.0 = perfectly balanced)")


def bench():
    """Rough forward+backward throughput for the layer alone.
    Numbers are machine-specific -- use them for order of magnitude."""
    print("\nBenchmark -- MoE layer fwd+bwd (d_model=512, 8 experts, top-2):")
    cfg = MoEConfig(d_model=512, ffn_hidden=384, n_experts=8, top_k=2)
    moe = MoELayer(cfg)
    for n_tokens in (1024, 8192):
        x = torch.randn(n_tokens // 256, 256, 512)
        # warmup
        for _ in range(2):
            moe(x).output.sum().backward()
            moe.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        steps = 5
        for _ in range(steps):
            out = moe(x)
            (out.output.sum() + out.aux_loss).backward()
            moe.zero_grad(set_to_none=True)
        dt = (time.perf_counter() - t0) / steps
        print(f"  {n_tokens:5d} tokens/step: {dt * 1000:7.1f} ms  -> "
              f"{n_tokens / dt:8.0f} tokens/sec")


if __name__ == "__main__":
    print("Running Stage-1 MoE tests:")
    test_shapes()
    test_dispatch_matches_reference()
    test_aux_loss_balanced_at_init()
    test_aux_loss_detects_collapse()
    test_gradients()
    demo_utilization()
    bench()
    print("\nAll tests passed.")
