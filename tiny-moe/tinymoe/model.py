"""The full decoder LM: token embeddings + N x TransformerBlock + tied head.

Design decisions:

* Tied embeddings: the output head SHARES the token embedding matrix
  (F.linear(x, embed.weight)). Saves vocab*d_model params (~2.1M here --
  6% of the model) and is standard for small LMs.

* One shared RotaryEmbedding for all layers (position info is not
  layer-specific). Its table is extensible via rope.extend() -- the Phase-2
  context-extension hook. The config's rope_scaling_type / rope_scaling_factor
  are the other hook: "none" | "linear" (PI) | "ntk", all handled in rope.py.

* Incremental KV cache: forward() accepts and returns per-layer caches, so
  generate() never recomputes the prefix. Shape per layer: 2 x
  (B, n_kv_heads, T, head_dim) -- small *because of GQA*.

* dense_ffn=True builds the param-matched dense baseline (see blocks.py).

* Forward output is a NamedTuple: logits, loss, aux_loss (mean over layers,
  unweighted), per-layer expert counts (for utilization logging).
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import TransformerBlock, RMSNorm
from .rope import RotaryEmbedding


@dataclass
class ModelConfig:
    # --- architecture ---
    vocab_size: int = 4096
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8            # query heads
    n_kv_heads: int = 2         # KV heads (GQA: 2 heads shared by 8 queries)
    head_dim: int = 64          # -> q width 8*64 = 512 = d_model
    max_seq_len: int = 2048     # training context (Phase 2: 4096 / 8192)
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    # --- RoPE (Phase-2 extension hooks) ---
    rope_base: float = 10000.0
    rope_scaling_type: str = "none"   # "none" | "linear" (PI) | "ntk"
    rope_scaling_factor: float = 1.0  # e.g. 4.0 for 2k -> 8k

    # --- FFN: MoE or dense baseline ---
    dense_ffn: bool = False     # True -> one dense SwiGLU FFN per block
    n_experts: int = 8
    top_k: int = 2
    ffn_hidden: int = 384       # per expert (MoE); for the dense baseline set
                                # this to n_experts * 384 = 3072 for
                                # parameter parity (see tests/test_model.py)

    aux_loss_coef: float = 0.01


class ModelOutput(NamedTuple):
    logits: torch.Tensor                  # (B, T, vocab)
    loss: Optional[torch.Tensor]          # scalar LM loss (if targets given)
    aux_loss: torch.Tensor                # scalar, mean over MoE layers (~1.0 = balanced)
    expert_counts: Optional[List[torch.Tensor]]  # per layer, (n_experts,)
    cache: Optional[list]                 # per-layer (k, v) for incremental decode


class TinyMoE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(
            cfg.head_dim, cfg.max_seq_len,
            base=cfg.rope_base,
            scaling_type=cfg.rope_scaling_type,
            scaling_factor=cfg.rope_scaling_factor,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

        self.apply(self._init_weights)
        # GPT-2 trick: the sublayer output projections (attn.wo, FFN w2) feed
        # straight into the residual stream, so scale their init std down by
        # 1/sqrt(2*n_layers) -- keeps residual magnitudes stable with depth.
        std = 0.02 / (2 * cfg.n_layers) ** 0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.wo.weight, std=std)
            if cfg.dense_ffn:
                nn.init.normal_(block.ffn.ffn.w2.weight, std=std)
            else:
                for expert in block.ffn.experts:
                    nn.init.normal_(expert.w2.weight, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    def forward(self, idx: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                cache: Optional[list] = None,
                use_checkpoint: bool = False) -> ModelOutput:
        """idx: (B, T) token ids. targets: (B, T) or None.
        cache: list of per-layer (k, v) or None. use_checkpoint: gradient
        checkpointing -- recompute block activations in backward to trade
        ~30% time for a large memory cut (needed for 2k+ training on 16GB)."""
        B, T = idx.shape

        # Position offset: with a cache, this segment continues after it.
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[2]
        cos, sin = self.rope(offset, T)

        x = self.embed(idx)
        aux_losses, counts_list = [], []
        # Return a cache when decoding incrementally (cache given) or when
        # generating (eval mode). During training we skip it -- it would just
        # hold memory.
        want_cache = cache is not None or not self.training
        new_cache = [] if want_cache else None

        for i, block in enumerate(self.blocks):
            cache_i = cache[i] if cache is not None else None
            if use_checkpoint and self.training:
                x, cache_out, aux, counts = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, cache_i, use_reentrant=False)
            else:
                x, cache_out, aux, counts = block(x, cos, sin, cache_i)
            aux_losses.append(aux)
            counts_list.append(counts)
            if new_cache is not None:
                new_cache.append(cache_out)

        x = self.final_norm(x)
        # Tied embedding head: same matrix as the input embedding.
        logits = F.linear(x, self.embed.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))

        aux_loss = torch.stack(aux_losses).mean()
        expert_counts = (counts_list if counts_list[0] is not None else None)
        return ModelOutput(logits, loss, aux_loss, expert_counts, new_cache)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: Optional[int] = 50,
                 cache: Optional[list] = None):
        """Autoregressive generation with the incremental KV cache.

        Phase 1 of a call = prefill: process the whole prompt in one forward
        (O(T^2) attention, but one matmul batch). Phase 2 = decode: one token
        at a time against the cache (O(T) per token). This is exactly the
        TTFT vs decode split the Phase-2 benchmark measures.
        """
        self.eval()
        for _ in range(max_new_tokens):
            if cache is None:
                out = self(idx, cache=None)
                cache = out.cache
            else:
                out = self(idx[:, -1:], cache=cache)   # only the new token
                cache = out.cache
            logits = out.logits[:, -1, :]              # (B, vocab)

            if temperature <= 0:                       # greedy
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx

    # ------------------------------------------------------------------
    def num_params(self, active: bool = False) -> int:
        """active=True counts only what one token's forward touches
        (top_k of n_experts FFNs) -- the MoE selling point."""
        if active and not self.cfg.dense_ffn:
            n = 0
            for name, p in self.named_parameters():
                if ".experts." in name:
                    # count only top_k experts' worth of expert weights
                    n += p.numel() // self.cfg.n_experts * self.cfg.top_k
                else:
                    n += p.numel()
            return n
        return sum(p.numel() for p in self.parameters())

    def kv_cache_bytes(self, seq_len: int, dtype_bytes: int = 4) -> int:
        """KV cache memory for `seq_len` tokens, batch 1."""
        per_token = (self.cfg.n_layers * 2 * self.cfg.n_kv_heads
                     * self.cfg.head_dim * dtype_bytes)
        return per_token * seq_len
