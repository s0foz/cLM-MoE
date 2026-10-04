"""Transformer block: RMSNorm + GQA attention + MoE (or dense) FFN, pre-norm.

Pre-norm layout (norm BEFORE the sublayer, residual around it):

    x = x + Attention(RMSNorm(x))
    x = x + MoE(RMSNorm(x))

Why pre-norm instead of post-norm: residual streams stay clean at
initialization (each sublayer starts as identity), which makes deep small
models train without warmup tricks. Modern LLMs (Llama, GPT-NeoX) all use it.

Why RMSNorm instead of LayerNorm: same stabilizing effect, minus the mean
subtraction -- one less reduction, cheaper on CPU, and empirically equal
quality. Formula: x * rsqrt(mean(x^2) + eps) * learned_weight.
"""

import torch
import torch.nn as nn

from .attention import Attention
from .moe import MoEConfig, MoELayer, MoEOutput, ExpertFFN


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learned scale

    def forward(self, x):
        # Compute the norm in fp32 (avoids overflow when we later train fp16),
        # then cast back to the input dtype.
        x_f = x.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * self.weight.float()).to(x.dtype)


class DenseFFN(nn.Module):
    """A single dense SwiGLU FFN behind the MoE interface.

    Exists so the *baseline comparison* model (dense, same total params)
    differs from the MoE model ONLY in the FFN -- same attention, same norms,
    same tokenizer. Anything else would contaminate the benchmark.
    forward() returns the same MoEOutput named tuple as MoELayer (aux loss 0,
    no routing stats) so TransformerBlock is branch-free.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ffn = ExpertFFN(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x) -> MoEOutput:
        return MoEOutput(self.ffn(x), x.new_zeros(()), None)


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        if cfg.dense_ffn:
            self.ffn = DenseFFN(cfg)
        else:
            self.ffn = MoELayer(MoEConfig(
                d_model=cfg.d_model,
                ffn_hidden=cfg.ffn_hidden,
                n_experts=cfg.n_experts,
                top_k=cfg.top_k,
                aux_loss_coef=cfg.aux_loss_coef,
            ))

    def forward(self, x, cos, sin, cache=None):
        """Returns (x, new_cache, aux_loss, expert_counts)."""
        a, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + a
        m = self.ffn(self.ffn_norm(x))
        x = x + m.output
        return x, new_cache, m.aux_loss, m.expert_counts
