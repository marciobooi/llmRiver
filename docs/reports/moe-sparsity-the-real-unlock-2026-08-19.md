# MoE: a 30B model that runs 2x faster than a 7B

2026-08-19, `mercury-hetzner` (AMD Ryzen 5 3600, 6c/12t, 44 GB/s measured
RAM bandwidth). Stock upstream llama.cpp `b10499`, isolated Docker,
`-t 6 -fa 1`, 128 tokens generated, `--ignore-eos`.

## The result

| model | params | file size | decode tok/s |
|---|---|---|---|
| Qwen2.5-7B-Instruct Q4_K_M (dense) | 7.6B | 4.68 GB | 9.08 |
| Qwen3-30B-A3B-Instruct-2507 Q4_K_M (MoE) | 30.5B total / ~3.3B active | 18.56 GB | **18.66** |

**4x the parameters. 4x the file size. 2.05x the speed.** And a
materially better model.

Prompt processing likewise: 62-68 tok/s vs. the dense model's 41 tok/s.

## Why — this is §I.3 of the critical-analysis doc, measured

> "A saída não é mover bytes mais depressa. É NÃO MOVER bytes que não
> são precisos. Ou seja: esparsidade."

A Mixture-of-Experts model routes each token to a small subset of its
experts. Total parameters sit in RAM; only the active ones are read per
token. Decode speed is set by *bytes read per token*, not by parameter
count:

```
44 GB/s / 18.66 tok/s = 2.38 GB read per token
```

vs. the dense 7B's full 4.68 GB. Roughly 16 GB of the MoE's weights are
resident in RAM and simply never touched on any given token. The model
is 4x larger and moves half the bytes.

This is the only change tested in this entire project that beat the
bandwidth wall rather than negotiating with it. Every other technique —
thread tuning, flash attention, huge pages, quantization, speculation,
batching, pinning — either shuffles the same bytes more cleverly or
amortizes them. Sparsity removes them.

## Two anti-synergies, both explained by one mechanism

MoE's advantage comes from *per-token* expert sparsity. Anything that
processes several tokens in one forward pass activates the **union** of
those tokens' expert sets, and the advantage erodes.

### 1. Speculative decoding actively hurts MoE

| config | tok/s |
|---|---|
| MoE alone | **18.5** |
| MoE + draft depth 2 | 13.3 |
| MoE + draft depth 3 | 11.6 |
| MoE + draft depth 5 | 10.0 |

Speculative decoding's founding assumption is "verifying N draft tokens
reads the same weights as verifying 1." That is true for a dense model
and **false for MoE** — N tokens route to N different expert sets, so
verification reads far more weight data than a single token would.
Speculation destroys the exact property that makes MoE fast. Worse with
depth, monotonically.

(For reference, on the dense 7B speculation *helped*: 9.0 -> 13.1.)

### 2. MoE batches worse than dense — but still wins

| batch | dense 7B | MoE 30B | dense scaling | MoE scaling |
|---|---|---|---|---|
| 1 | 9.08 | 18.66 | 1.00x | 1.00x |
| 4 | 25.13 | 33.27 | 2.77x | 1.78x |
| 16 | 33.98 | **45.53** | 3.74x | 2.44x |

The dense model amortizes batching almost ideally (3.74x) because every
sequence in the batch reads the *same* weights. The MoE only reaches
2.44x because each added sequence pulls in more experts, so
bytes-per-pass grows with batch size instead of staying flat.

Its lead therefore shrinks — 2.05x at batch 1, 1.34x at batch 16 — but
it is still the fastest option at every batch size tested.

## Best configurations found

| use case | configuration | tok/s | vs. session baseline |
|---|---|---|---|
| single user, lowest latency | MoE 30B, no speculation | **18.66** | 2.05x |
| many concurrent users | MoE 30B at batch 16 | **45.53** | 5.02x |

Session baseline = dense Qwen2.5-7B Q4_K_M, tuned, single stream
(9.08 tok/s).

Explicitly **not** recommended with MoE: speculative decoding (hurts),
and do not assume the dense model's tuning transfers.

## Methodological note: the roofline profiler measured the wrong thing

Step 1's `tools/roofline` reported a 14.9% latency improvement from 2MB
huge pages, measured with a **random pointer-chase**. That gain never
materialized for real inference (measured: 9.0 -> 9.1 tok/s, noise, even
with THP forced to `always` system-wide and `--no-mmap` to put weights
in THP-eligible anonymous memory).

The reason is that LLM decode streams weights **sequentially**.
Sequential access already has near-ideal TLB behavior — one TLB miss per
4KB page while doing 4KB of useful work. Huge pages help *random*
access, which is what the profiler benchmarked and is not what this
workload does.

Lesson for the resistance-matrix idea: a resistance number is only
transferable if it was measured with the access pattern the real
workload uses. The profiler should measure sequential streaming
explicitly, not just pointer-chase latency.
