# Architecture comparison: which model is fastest depends on context length

2026-08-19, `mercury-hetzner` (AMD Ryzen 5 3600, 6 physical cores / 12
SMT threads, 44 GB/s measured RAM bandwidth). llama.cpp b10502,
`llama-bench -p 0 -n 32 -d 0,4096,16384 -t 6 -fa 1`, isolated Docker.

Motivated by the finding (see PLAN.md Attempt 15) that the long-context
bottleneck on this host is **attention compute**, not weight bandwidth
and not KV bandwidth — so the remaining levers are architectural.

## Results — decode tok/s by KV depth

| model | architecture | size | 0 | 4,096 | 16,384 | decay |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B Q2_K | MoE, full attn, 48L | 11.3 GB | **26.77** | 10.25 | 2.34 | -91% |
| Qwen3-30B-A3B Q4_K_M | MoE, full attn, 48L | 18.6 GB | 19.58 | 9.20 | — | — |
| DeepSeek-V2-Lite Q4_K_M | **MoE + MLA** | 10.4 GB | 21.16 | **12.22** | **3.76** | -82% |
| Gemma 3 12B Q4_K_M | dense, **sliding-window attn** | 7.3 GB | 4.97 | 4.58 | 3.12 | **-37%** |
| Muse Glimmer 30B Q4_K_M | **dense**, full attn | 16.8 GB | 2.54 | 2.31 | 2.12 | -17% |
| Qwen2.5-7B Q4_K_M | dense, full attn | 4.7 GB | 9.30 | 7.24 | — | — |

## What each architecture actually bought

**MoE (expert sparsity) — solves the short-prompt wall.** Qwen3-30B-A3B
reads ~2.4 GB per token out of 18.6 GB of weights, so it beats a dense
7B by 2x while being 4x larger. But sparsity applies only to FFN
weights; attention is dense, and this model carries 48 attention layers.
It therefore has the *worst* context decay measured (-91%).

**MLA (compressed KV) — the best all-rounder.** DeepSeek-V2-Lite is a
MoE *and* uses Multi-head Latent Attention, which projects K/V into a
low-rank latent instead of caching them in full. It attacks both
bottlenecks at once: sparsity for weight bandwidth, latent KV for
attention cost. Result: **fastest at 4K and fastest at 16K**, and within
20% of the best short-prompt number. If one model had to be picked for
mixed workloads on this host, it is this one.

**Sliding-window attention — best decay, but needs a fast baseline.**
Gemma 3 caps per-token attention work at the window, giving the second
flattest curve (-37%) and overtaking the Qwen MoE past ~13K. But it is a
*dense* 12B, so its starting point is bandwidth-limited (4.97 tok/s,
against the 6.0 its 7.3 GB implies at 44 GB/s). Flat and slow only wins
late.

**Dense, whatever the attention scheme, is capped by its own size.**
Muse Glimmer has the flattest curve of all (-17%) and is still the
slowest model at every depth tested, because 44 GB/s / 2.54 tok/s =
17.3 GB read per token, i.e. its entire weight file, every token. No
attention optimization can compensate for reading everything.

## The general rule this establishes

Two independent properties, and a model needs both:

| property | fixes | measured by |
|---|---|---|
| **sparsity** (MoE) | the short-prompt / weight-bandwidth wall | tok/s at depth 0 |
| **cheap attention** (MLA, sliding window) | the long-context / attention-compute wall | decay from 0 to 16K |

- Qwen3-MoE has the first, not the second: fast start, worst decay.
- Gemma 3 has the second, not the first: slow start, good decay.
- Muse Glimmer (dense) has neither in a form that helps: slow throughout.
- **DeepSeek-V2-Lite has both**, and wins wherever context is non-trivial.

## Recommendation for this host

| workload | model | tok/s |
|---|---|---|
| short prompts only | Qwen3-30B-A3B Q2_K | 25.7 |
| mixed / realistic context | **DeepSeek-V2-Lite Q4_K_M** | 21.2 short, 12.2 at 4K, 3.8 at 16K |
| very long context (>13K) | DeepSeek-V2-Lite, else Gemma 3 | 3.8 / 3.1 |
| many concurrent users | Qwen3-30B-A3B Q4_K_M, batch 16 | 45.5 aggregate |

## Caveats

- These are architectures *as shipped*, not a controlled experiment.
  The models differ in parameter count, quantization, training data,
  layer count and release date, not only in attention scheme. The decay
  *slopes* are the transferable signal; absolute values confound several
  variables.
- **Quality was not measured.** DeepSeek-V2-Lite (16B, 2024) is a
  smaller and older model than Qwen3-30B-A3B (2025); winning on speed
  does not make it the better answer. Anyone acting on this should
  evaluate output quality on their own task before switching.
- At 16K context every model here runs 2-4 tok/s. Sliding-window and
  MLA change the slope; they do not make long-context CPU inference on
  a 6-core desktop part comfortable.
