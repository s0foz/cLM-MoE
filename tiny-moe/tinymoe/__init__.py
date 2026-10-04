"""tiny-moe: a small Mixture-of-Experts decoder LM, built for learning.

Stage 1: MoE layer -- router, top-k gating, load-balancing loss.
Stage 2: full decoder -- RoPE, GQA attention, blocks, model, KV cache.
"""
from .moe import MoEConfig, MoELayer, MoEOutput, TopKRouter, ExpertFFN
from .rope import RotaryEmbedding, build_rope_cache, apply_rope
from .attention import Attention
from .blocks import TransformerBlock, RMSNorm, DenseFFN
from .model import ModelConfig, ModelOutput, TinyMoE

__all__ = [
    "MoEConfig", "MoELayer", "MoEOutput", "TopKRouter", "ExpertFFN",
    "RotaryEmbedding", "build_rope_cache", "apply_rope",
    "Attention", "TransformerBlock", "RMSNorm", "DenseFFN",
    "ModelConfig", "ModelOutput", "TinyMoE",
]
