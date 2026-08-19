# Batching, and the roofline crossover that explains every result so far

2026-08-19, `mercury-hetzner` (AMD Ryzen 5 3600, 6c/12t, 2x32GB DDR4-3200
dual-channel), isolated Docker, stock upstream llama.cpp `b10499`
(`llama-batched-bench`, `-t 6 -fa 1`, 64-token prompt, 128 tokens
generated). Raw logs: `llamacpp-batched-q4km-2026-08-19.log`,
`llamacpp-batched-q3km-2026-08-19.log`.

## 1. Batching: 3.7x aggregate throughput, then a hard wall

Qwen2.5-7B-Instruct Q4_K_M, decode (token-generation) throughput
aggregated across all parallel sequences:

| parallel sequences | decode tok/s | vs. batch=1 |
|---|---|---|
| 1 | 9.08 | 1.00x |
| 2 | 14.82 | 1.63x |
| 4 | 25.13 | 2.77x |
| 8 | 26.44 | 2.91x |
| 16 | **33.98** | **3.74x** |
| 32 | 33.81 | 3.72x |
| 64 | 33.89 | 3.73x |

Dead flat from 16 onward — 33.98 / 33.81 / 33.89. Adding sequences past
16 buys literally nothing.

This is §II.5 of the critical-analysis doc demonstrated directly: at
batch=1 arithmetic intensity is ~1 MAC/byte, because producing one token
requires reading every weight. Each additional concurrent sequence rides
along on bytes *already paid for* — the weights are read once per
forward pass regardless of how many sequences are in it. Batching is an
impedance transformer in exactly the doc's sense: it converts a
low-intensity workload into a high-intensity one without touching the
hardware.

The plateau at 16 is the transformation completing. The system has left
the bandwidth-bound region entirely and hit the Ryzen's AVX2 FMA
throughput instead. Past that point more parallelism has nothing left
to amortize.

**Caveat, stated plainly:** this is *aggregate* throughput, not
single-stream latency. At B=16 each individual sequence sees ~2.1 tok/s,
much slower than the 9.08 tok/s a lone request gets. Batching is a win
for serving many concurrent users; it is a loss for one user waiting on
one answer.

## 2. The inversion: optimal quantization depends on the regime

Same machine, same flags, Q3_K_M (3.80GB) vs. Q4_K_M (4.68GB):

| batch | Q4_K_M | Q3_K_M | winner |
|---|---|---|---|
| 1 | 9.08 | **10.94** | Q3_K_M, +20% |
| 4 | **25.13** | 20.71 | Q4_K_M, +21% |
| 16 | **33.98** | 23.44 | Q4_K_M, +45% |

A complete reversal, and the margin *grows* with batch size.

- **Bandwidth-bound (batch=1):** fewer bytes wins. Q3_K_M moves ~19%
  less data per forward pass and converts that almost 1:1 into tokens.
- **Compute-bound (batch>=4):** more dequantization work loses. K-quant
  unpacking costs ALU work per weight, and once bytes are amortized
  across a batch, that unpack cost is the thing standing between you and
  the answer. Q3_K_M's more elaborate unpacking makes it strictly worse.

## 3. The same model explains a result from earlier in the day

Prompt processing was *always* faster on Q4_K_M, in every single test:
41.6 vs. 27.0 tok/s here, 38.8 vs. 26.9 tok/s in the earlier
single-stream `llama-cli` runs. That looked like a curiosity at the time.

It is the same effect. Prompt processing is inherently batched — it
consumes many tokens in one pass — so it has *always* been in the
compute-bound region, even for a single user. Q3_K_M was never winning
there.

One roofline model now accounts for every measurement in this session:

| workload | regime | faster quant |
|---|---|---|
| prompt processing (any batch) | compute-bound | Q4_K_M |
| decode, batch=1 | bandwidth-bound | Q3_K_M |
| decode, batch>=4 | compute-bound | Q4_K_M |

## 4. Practical guidance for this host

- **Serving concurrent users:** Q4_K_M at batch 16. ~34 tok/s aggregate,
  3.7x the naive single-stream configuration, with *better* quality than
  Q3_K_M. The quality/speed tradeoff that quantization forces at batch=1
  simply disappears here.
- **One user, lowest latency:** Q3_K_M with speculative decoding, or
  Q4_K_M + speculative decoding (13.1 tok/s on predictable text). Do not
  batch.
- **Do not** assume a quantization choice transfers between these two
  cases. It inverts.

## 5. What this says about the project's own roadmap

Step 2's candidate claim (compute-aware weight reordering + credit-based
prefetch, >20% on a memory-constrained model) targets the
*bandwidth-bound* region. That region is real but narrow: it is
single-stream decode only, and it is already within 3% of its physical
ceiling on this host (44 GB/s measured / 4.68GB model = ~9.4 tok/s;
llama.cpp achieves 9.15).

Anything that amortizes weight reads — batching, speculative decoding —
moves the system out of that region entirely, and then bandwidth
optimizations have nothing left to optimize. This does not invalidate
the reordering idea, but it bounds it sharply: **it can only ever help
the batch=1 case, and only up to ~3% on this machine.**
