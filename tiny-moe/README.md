# tiny-moe

A tiny Mixture-of-Experts decoder LM (10–50M params), built from scratch to
understand the architecture, then quantized and run on CPU.

## Project structure (target)

```
tiny-moe/
├── README.md
├── requirements.txt
├── configs/
│   └── tiny.yaml             # model + training hyperparameters      [stage 3]
├── tinymoe/
│   ├── __init__.py
│   ├── moe.py                # ✅ stage 1: router + experts + aux loss
│   ├── rope.py                ✅ stage 2: RoPE + PI/NTK scaling hook
│   ├── attention.py           ✅ stage 2: GQA + incremental KV cache
│   ├── blocks.py              ✅ stage 2: RMSNorm block + dense baseline
│   ├── model.py               ✅ stage 2: full decoder + generate()
│   ├── tokenizer.py          # small BPE tokenizer                   [stage 3]
│   ├── data.py               # TinyStories / WikiText-2 loading      [stage 3]
│   ├── train.py              # training loop + logging + checkpoints [stage 3]
│   ├── export/
│   │   ├── gguf_writer.py    # PyTorch -> GGUF conversion            [stage 5]
│   │   └── numpy_quant.py    # numpy int8/int4 quantization          [stage 5]
│   └── infer_cpu.py          # CPU generation from quantized model   [stage 6]
├── tests/
│   ├── test_moe.py            ✅ stage 1: MoE correctness + demo
│   └── test_model.py          ✅ stage 2: causality, cache eq, GQA, RoPE
└── checkpoints/              # saved .pt files                       [stage 3]
```

## Build stages

1. **MoE layer** — router (softmax gating, top-2), SwiGLU experts,
   Switch-style load-balancing loss. ✅
2. **Transformer** — RoPE, GQA attention, pre-norm blocks, full model.
3. **Training** — tokenizer, dataset, mixed-precision-capable loop, logging
   (loss / expert utilization / tokens-sec), checkpointing.
4. **Training run** — small run on TinyStories.
5. **Quantization** — own GGUF writer + numpy int8/int4 fallback.
6. **CPU inference** — generate text from the quantized model.

## Design decisions (summary)

- **SwiGLU experts, 3 matrices each** — standard for modern LLMs, small
  quality win over GELU MLP.
- **Top-2 of 8 experts** — 25% of FFN params active per token; total params
  can be large while FLOPs/token stay small (the MoE selling point).
- **Renormalized gates** — chosen gate weights sum to 1 so output scale is
  independent of k.
- **Switch aux loss** `L = E * Σ f_i·P_i`, min 1.0 when balanced — hard usage
  counts (no grad) + soft mean probabilities (grad). Coef 0.01.
- **Sorted dispatch** (grouped GEMM in a loop over 8 experts) instead of
  dense all-experts compute — 8x fewer FLOPs, no exotic kernels.
- **GQA (stage 2)** — 8 query heads / 2 KV heads keeps the KV cache 4x
  smaller than MHA at 2048 context.
- **No capacity factor / token dropping** — unnecessary at this scale.

## Phase-2 optimization backlog

Known inefficiencies to fix when we do the KV-cache engineering stage:

1. **KV cache grows via `torch.cat` per generated token** — an O(T) copy per
   token (O(T^2) memcpy over a whole generation). Fine at 8k/our scale, but
   the right fix is a preallocated buffer with capacity doubling (amortized
   O(1) appends) or a ring buffer. Keep the `torch.cat` version as the
   reference implementation for correctness testing.
2. **8-bit KV cache** — compress cache to int8 + per-head scale; compare
   memory vs generation quality.
3. **GQA `repeat_interleave` per forward** — expand-in-place / view tricks if
   the benchmark shows it mattering.
