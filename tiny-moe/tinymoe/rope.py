"""Rotary Position Embeddings (RoPE) -- with a pluggable scaling hook.

What RoPE does
--------------
For every pair of channels in a query/key head, rotate it by an angle that
depends on the token's absolute position:

    angle(pos, i) = pos * theta_i ,   theta_i = base^(-2i/d)   (d = head_dim)

The magic property: the dot product between a query at position m and a key at
position n depends only on (m - n) -- rotations compose by adding angles.
So RoPE gives you *relative* position information through *absolute* rotations,
with no learned position table.

Why scale it at all (Phase 2)
-----------------------------
A model trained at 2048 has never seen the rotation phase differences that
appear at 4k/8k positions. Both scaling methods below remap the new, unseen
positions into a range the model has seen:

  * "linear" (Position Interpolation, PI):
        angle(pos) = (pos / s) * theta_i
    Simply divide positions by the extension factor s (e.g. 4 for 2k->8k).
    Every 8k position looks like a position <= 2048 to the model. Cheap and
    reliable, but the model must fine-tune to adjust (positions are now
    "compressed", nearby tokens get similar rotations).

  * "ntk" (NTK-aware scaling):
        base' = base * s^(d / (d-2))
    Inflate the RoPE base instead of shrinking positions. High-frequency
    components (which carry local position info) get stretched a lot, while
    low-frequency components (which carry long-range info) barely change.
    Often works with little or no fine-tuning.

We implement both behind one config switch so Phase 2 can A/B them without
touching model code. (YaRN exists and is better still -- skipped for now.)

Buffers are build-once and extensible: `extend()` lets fine-tuning push the
cache from 2k to 4k/8k without rebuilding the model.
"""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dim in half-split convention (Llama-style):
    [x1, x2] -> [-x2, x1]. Paired with cos/sin repeated across the halves."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotation to q and k. Shapes: q/k (B, H, T, d), cos/sin (T, d)."""
    # (T, d) -> (1, 1, T, d) for broadcasting over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0,
                     scaling_type: str = "none", scaling_factor: float = 1.0,
                     device=None, dtype: torch.dtype = torch.float32):
    """Build (cos, sin) tables of shape (seq_len, head_dim).

    Table layout: column j of the table matches channel j of the head under
    the half-split convention -- angles are duplicated across both halves so
    they broadcast cleanly against q/k in apply_rope().
    """
    assert scaling_type in ("none", "linear", "ntk"), scaling_type
    half = head_dim // 2

    if scaling_type == "ntk":
        # Base inflation (static NTK variant). "Dynamic" NTK (scale the base
        # by the *current* sequence length once it exceeds training length) is
        # a small extension of this function -- left for Phase 2 experiments.
        base = base * scaling_factor ** (head_dim / (head_dim - 2))

    # theta_i = base^(-2i/d) for i in 0..d/2-1
    i = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = base ** (-i / half)                     # (half,)

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    if scaling_type == "linear":
        positions = positions / scaling_factor          # position interpolation

    angles = positions[:, None] * inv_freq[None, :]     # (seq_len, half)
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(dtype)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(dtype)
    return cos, sin


class RotaryEmbedding(nn.Module):
    """Owns the (cos, sin) table for one model. Shared across all layers --
    it is position info, not layer-specific state."""

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0,
                 scaling_type: str = "none", scaling_factor: float = 1.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        cos, sin = build_rope_cache(max_seq_len, head_dim, base,
                                    scaling_type, scaling_factor)
        # persistent=False: pure derived tables -- don't bloat checkpoints.
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def max_seq_len(self) -> int:
        return self.cos.shape[0]

    def extend(self, new_max_seq_len: int):
        """Grow the table for longer contexts (Phase 2 fine-tuning at 4k/8k).
        Cheap to call repeatedly; rebuilds only when actually growing."""
        if new_max_seq_len > self.max_seq_len:
            cos, sin = build_rope_cache(new_max_seq_len, self.head_dim,
                                        self.base, self.scaling_type,
                                        self.scaling_factor,
                                        device=self.cos.device,
                                        dtype=self.cos.dtype)
            self.cos = cos       # re-register as plain attributes
            self.sin = sin

    def forward(self, offset: int, seq_len: int):
        """Return the (cos, sin) slice for positions
        [offset, offset + seq_len). offset > 0 when appending to a KV cache."""
        assert offset + seq_len <= self.max_seq_len, (
            f"positions {offset}+{seq_len} exceed cache {self.max_seq_len}; "
            f"call rope.extend() first")
        return self.cos[offset:offset + seq_len], self.sin[offset:offset + seq_len]
