"""Grouped-query attention (GQA) with an incremental KV cache.

GQA in one sentence
-------------------
Standard multi-head attention (MHA) has n_heads independent K/V heads.
Multi-query attention (MQA) has ONE shared K/V head. GQA is the middle
ground: n_kv_heads K/V heads, each shared by a *group* of query heads
(here: 8 query heads / 2 KV heads -> groups of 4).

Why it matters: the KV cache stores K and V for every token forever.
With 2 KV heads instead of 8 the cache is 4x smaller -- see the memory
table in tests/test_model.py. That is the whole trick; quality loss
vs MHA is small at this scale (GQA paper, Ainslie et al. 2023).

Incremental KV cache
--------------------
During generation, token t's query only needs to attend to keys/values
1..t. Caching them means each new token costs ONE attention call over the
prefix (O(T) per token) instead of re-running attention over the whole
sequence (O(T^2) per token). The cache is passed in and returned updated --
this module holds no state of its own, which keeps it testable.

Cache format: (k, v) tensors of shape (B, n_kv_heads, T_cached, head_dim).
K/V in the cache are already RoPE-rotated (positions are baked in), so we
only rotate the *new* keys -- standard practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        assert cfg.n_heads % cfg.n_kv_heads == 0, \
            "query heads must be divisible by KV heads"
        self.group_size = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        """x: (B, T, d). cos/sin: (T, head_dim) for THIS segment's positions.
        cache: optional (k, v) with earlier tokens. Returns (out, new_cache)."""
        B, T, _ = x.shape

        # Project to heads: (B, H, T, hd)
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Rotate q and the NEW keys only (cached keys are already rotated).
        q, k = apply_rope(q, k, cos, sin)

        # Append to cache -> full K/V for the prefix.
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)   # (B, n_kv, T_cached+T, hd)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        K = k.shape[2]

        # GQA expansion: repeat each KV head for its group of query heads.
        # (B, n_kv, K, hd) -> (B, n_heads, K, hd). Done AFTER rope/cache so
        # the cache stays small -- only the compute path sees full width.
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        # Causal mask. Three cases:
        #   T == 1          : decoding one token -> it may see everything up
        #                     to K. No mask needed (cheapest path).
        #   K == T          : prefill from scratch -> plain lower triangle.
        #   otherwise       : chunked append; query i (of T) may see keys
        #                     j <= K - T + i. tril(diagonal=K-T) encodes this.
        if T == 1:
            attn_mask = None
        else:
            attn_mask = torch.ones(T, K, dtype=torch.bool,
                                   device=x.device).tril(diagonal=K - T)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Merge heads and project back: (B, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out), new_cache
