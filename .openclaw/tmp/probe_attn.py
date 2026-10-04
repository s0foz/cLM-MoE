"""Probe: cost of attention at long context on CPU (this sandbox: 3 vCPU Xeon).
Scaled model dims: 8 query heads, head_dim=64, batch=1, fp32."""
import time, resource, torch
import torch.nn.functional as F

torch.set_num_threads(3)
B, H, D = 1, 8, 64

def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

for T in (512, 2048, 8192):
    q = torch.randn(B, H, T, D, requires_grad=True)
    k = torch.randn(B, H, T, D, requires_grad=True)
    v = torch.randn(B, H, T, D, requires_grad=True)
    scores_mem = B * H * T * T * 4 / 1e6   # fp32 score matrix, one layer
    # warmup
    F.scaled_dot_product_attention(q, k, v).sum().backward()
    t0 = time.perf_counter()
    out = F.scaled_dot_product_attention(q, k, v)
    out.sum().backward()
    dt = time.perf_counter() - t0
    # per-layer cost x6 layers, fwd+bwd ~2.5x fwd
    print(f"T={T:5d}  fwd+bwd 1 layer: {dt*1000:8.1f} ms | "
          f"score matrix/layer: {scores_mem:7.1f} MB | "
          f"x6 layers: {scores_mem*6:7.1f} MB | peak RSS: {rss_mb():7.0f} MB | "
          f"est 6-layer attn only: {dt*6*2.5:6.2f} s/step")
