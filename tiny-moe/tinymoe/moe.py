"""Stage 1: the Mixture-of-Experts feed-forward layer.

Three pieces live in this file:

  1. TopKRouter  -- softmax gating over experts, picks top-k per token,
                    and computes the load-balancing auxiliary loss.
  2. ExpertFFN   -- one expert = a small SwiGLU feed-forward network.
  3. MoELayer    -- routes tokens to experts and recombines the results.

Notation used in the docstrings:
  B = batch size, T = sequence length, d = d_model,
  E = number of experts, k = experts active per token,
  N = B*T = total number of tokens in the batch.

Design decisions (the "why"):
  * Gates are renormalized over the chosen experts so the output scale does
    not depend on k. Without this, k=2 would roughly double the magnitude of
    the layer output compared to k=1.
  * The auxiliary loss follows Switch Transformer:  L = E * sum_i(f_i * P_i).
    It is minimized (= 1.0) exactly when every expert gets the same share of
    tokens. See TopKRouter._load_balancing_loss for details.
  * Dispatch is done by sorting (token, expert) pairs so each expert's work is
    one contiguous slice -> one matmul per expert, no fancy kernels needed.
    The naive alternative (run every expert on every token and mask) wastes
    E=8x the FLOPs and is noticeably slower on CPU.
  * No capacity factor / token dropping: at our scale every token can be
    served. Real MoEs cap the number of tokens per expert and drop the
    overflow; we deliberately skip that complexity.
"""

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class MoEConfig:
    """Hyperparameters for one MoE layer."""

    d_model: int = 512          # transformer width (input/output of the layer)
    ffn_hidden: int = 384       # hidden width of ONE expert
    n_experts: int = 8          # total experts
    top_k: int = 2              # experts used per token
    aux_loss_coef: float = 0.01 # weight of the aux loss in the training loss
    router_init_std: float = 0.02


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
class MoEOutput(NamedTuple):
    """What MoELayer.forward returns.

    aux_loss is returned UNWEIGHTED (its raw value, ~1.0 when balanced).
    The training loop multiplies it by MoEConfig.aux_loss_coef and adds it to
    the LM loss -- keeping the two loss terms separate makes them easy to log.
    """

    output: torch.Tensor         # (B, T, d_model)
    aux_loss: torch.Tensor       # scalar, raw load-balancing loss (min = 1.0)
    expert_counts: torch.Tensor  # (E,) int64: token-slots routed to each expert


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
class TopKRouter(nn.Module):
    """Softmax gating + top-k expert selection + load-balancing loss.

    For each token x (a vector of size d_model):

        logits = W_router @ x           # (E,) raw score per expert
        probs  = softmax(logits)        # (E,) routing probabilities
        pick the k experts with the largest probs,
        renormalize the k chosen probs so they sum to 1.

    Why softmax (and not, say, sigmoid per expert)? Because we want the gate
    weights for a token to form a distribution that sums to 1 -- it makes the
    renormalization well-defined and the aux loss interpretable as "shares".
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        # No bias: the bias would be a constant "expert preference" that the
        # aux loss would then fight against. Keep the router minimal.
        self.gate = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        # Small init -> near-uniform probs at start -> balanced routing at the
        # beginning of training, so experts all learn something before the
        # router starts to specialize.
        nn.init.normal_(self.gate.weight, std=cfg.router_init_std)

    def forward(self, x: torch.Tensor):
        """x: (N, d_model) -> top_idx (N,k), top_probs (N,k), aux_loss ()."""
        logits = self.gate(x)                        # (N, E)
        probs = F.softmax(logits, dim=-1)            # (N, E)

        top_probs, top_idx = probs.topk(self.top_k, dim=-1)   # (N, k) each

        # Renormalize the chosen gates to sum to 1 per token.
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        aux_loss = self._load_balancing_loss(probs, top_idx)
        return top_idx, top_probs, aux_loss

    def _load_balancing_loss(self, probs: torch.Tensor,
                             top_idx: torch.Tensor) -> torch.Tensor:
        """Switch Transformer auxiliary loss.

            L = E * sum_i f_i * P_i

        where
            f_i = fraction of routed token-slots that went to expert i
                  (a HARD count -- treated as a constant, no gradient),
            P_i = mean router probability assigned to expert i
                  (soft -- this is where the gradient flows).

        Why it works:
          * If one expert is overused, say f_1 is large -- then the term
            f_1 * P_1 is large unless the router *lowers* P_1 (its probability
            for that expert). Gradient descent does exactly that, pushing
            tokens toward the underused experts.
          * Minimum value: when everything is uniform, f_i = P_i = 1/E for all
            i, so L = E * E * (1/E)(1/E) = 1.0. So "aux loss == 1.0" is our
            health check for a balanced router, and anything clearly above
            (2.0+, or E in the worst case) means routing collapse.

        Useful range: minimum 1.0 (perfectly balanced). The *maximum* is
        E/k, not E: even a fully collapsed router must pick k distinct
        experts per token, so usage can never concentrate on fewer than k
        experts. For our config (E=8, k=2) that means aux in [1.0, 4.0],
        and > ~3 means only k experts are effectively alive.

        Why the hard count for f_i (instead of a soft fraction):
          The count is the *actual* dispatch decision -- that is the thing we
          want to balance. Making f_i differentiable too (e.g. f_i = mean
          probability) gives a weaker signal: both factors can shrink together
          instead of fixing the imbalance. The count acts as a fixed "blame
          assignment" and the probability gets the corrective gradient.

        Note on top-k: f_i counts all k*N dispatch slots, not just top-1.
        Variants exist (GShard balances per-expert *importance* instead);
        this slot-count version is the simplest one that works well.
        """
        N, E = probs.shape
        k = top_idx.shape[1]

        # Hard usage fraction f_i (no gradient through the counts).
        ones = torch.ones_like(top_idx, dtype=probs.dtype)
        counts = torch.zeros(E, device=probs.device,
                             dtype=probs.dtype).index_add_(
            0, top_idx.reshape(-1), ones.reshape(-1))     # (E,)
        f = counts / (N * k)                              # (E,)

        # Soft mean probability P_i (gradient flows here).
        P = probs.mean(dim=0)                             # (E,)

        return E * (f * P).sum()


# --------------------------------------------------------------------------
# One expert
# --------------------------------------------------------------------------
class ExpertFFN(nn.Module):
    """SwiGLU feed-forward network:  out = W2 @ (silu(W1 x) * W3 x).

    Why SwiGLU instead of the classic GELU MLP (W2 @ gelu(W1 x))?
    It is what Llama/Mistral-class models use and it consistently trains a bit
    better at the same parameter count. Cost: 3 weight matrices per expert
    instead of 2. If you ever want the simple version, it is a 2-line change
    (see forward's comment).

    Note ffn_hidden is the width of ONE expert. In dense models the FFN hidden
    is ~4x d_model; with MoE each expert is kept narrower (the ensemble makes
    up for it) so the total parameter count stays in budget.
    """

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_hidden, bias=False)  # gate proj
        self.w3 = nn.Linear(d_model, ffn_hidden, bias=False)  # up proj
        self.w2 = nn.Linear(ffn_hidden, d_model, bias=False)  # down proj
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU variant:  return self.w2(F.gelu(self.w1(x)))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------
# The MoE layer
# --------------------------------------------------------------------------
class MoELayer(nn.Module):
    """Top-k MoE feed-forward layer.

    forward(x): (B, T, d_model) -> MoEOutput with output of the same shape.

    Dispatch strategy ("grouped GEMM with a sort"):
      1. Flatten all tokens to (N, d).
      2. Router decides: each token gets k (expert, gate) pairs -> N*k pairs.
      3. Sort the pairs by expert id. After the sort, every expert's work is
         one contiguous slice of the arrays -> exactly one matmul per expert.
      4. Each expert processes its slice; results are weighted by the gate and
         accumulated back to the token rows with index_add_ (a token appears k
         times -> its k weighted expert outputs sum up, which is precisely the
         MoE formula  out(token) = sum_j gate_j * expert_j(token)).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = TopKRouter(cfg)
        self.experts = nn.ModuleList(
            [ExpertFFN(cfg.d_model, cfg.ffn_hidden)
             for _ in range(cfg.n_experts)])

    def forward(self, x: torch.Tensor) -> MoEOutput:
        B, T, d = x.shape
        N = B * T
        cfg = self.cfg
        E, k = cfg.n_experts, cfg.top_k

        flat = x.reshape(N, d)

        # --- routing -----------------------------------------------------
        top_idx, top_probs, aux_loss = self.router(flat)   # (N,k), (N,k), ()

        # --- build the (token, expert) pair list -------------------------
        # reshape(-1) is row-major: pairs come out token-major, i.e. the k
        # pairs of token 0 first, then token 1's, ... which is exactly what
        # repeat_interleave(k) produces for the token ids. Keep these two
        # aligned or dispatch will silently scramble the batch.
        pair_expert = top_idx.reshape(-1)                  # (N*k,)
        pair_gate = top_probs.reshape(-1)                  # (N*k,)
        pair_token = torch.arange(N, device=x.device).repeat_interleave(k)

        # --- sort by expert ---------------------------------------------
        order = torch.argsort(pair_expert, stable=True)    # (N*k,)
        pair_token_s = pair_token[order]
        pair_gate_s = pair_gate[order]
        counts = torch.bincount(pair_expert, minlength=E)  # (E,)
        assert int(counts.sum()) == N * k

        # --- per-expert compute + weighted recombination -----------------
        out = flat.new_zeros(N, d)
        start = 0
        for e, expert in enumerate(self.experts):
            n_e = int(counts[e])
            if n_e == 0:
                continue          # expert got no tokens this step
            tok = pair_token_s[start:start + n_e]          # (n_e,)
            gate = pair_gate_s[start:start + n_e, None]    # (n_e, 1)
            h = expert(flat[tok])                          # (n_e, d)
            # index_add_ accumulates: a token that appears twice (k=2) gets
            # gate1*expert_a(x) + gate2*expert_b(x).
            out.index_add_(0, tok, h * gate)
            start += n_e

        return MoEOutput(out.view(B, T, d), aux_loss, counts)
