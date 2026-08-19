# PLAN

Source of truth for the reasoning: [continuous-data-river-critical-analysis.md](continuous-data-river-critical-analysis.md).
This file is the actionable distillation — update it as steps complete.

## Guiding rule (§V.4 of the analysis)

> Before writing any runtime, build the roofline of ONE specific machine.
> Then pick ONE falsifiable claim and try to kill it. If it survives,
> there's a project. If not, you learned that in a week instead of a year.

Do not skip ahead to runtime/compiler work (§19/§36 territory — TVM/IREE/
tinygrad-scale, explicitly flagged in the analysis as the most dangerous
section because it's where a document quietly becomes undeliverable).

## Step 1 — Resistance-matrix profiler (`tools/roofline/`)

Status: **in progress**.

Measures the actual physical resistances of a target machine instead of
assuming them:

- [x] RAM bandwidth: single-thread and multi-thread streaming
      read/write/copy, to find the per-core ceiling and where it
      saturates (§III.1 — line fill buffers, not compute, are expected
      to be the wall around 4-6 cores).
- [x] Pointer-chase latency → implied outstanding-request concurrency
      via Little's Law (`bytes in flight = bandwidth × latency`),
      cross-checked against the ~12 LFB × 64B theoretical ceiling.
- [x] Disk sequential and random-4K throughput against a real file
      (O_DIRECT where available).
- [x] Compute throughput: AVX-512 FP32 FMA GFLOP/s and AVX-512 VNNI
      INT8 GOP/s (proxy for "GEMM throughput achievable in INT4/INT8" —
      exact INT4 unpack-and-MAC kernel is a later, separate step once a
      specific quantization format is chosen).
- [x] 4KB-page vs 2MB-hugepage random-access throughput, to quantify
      the TLB effect directly (§III.3.1).
- [ ] llama.cpp stall breakdown — needs an actual llama.cpp integration
      point; deferred until Step 1's standalone numbers justify it.

Explicitly out of scope for this tool: LFB PMU occupancy counters via
raw `perf_event_open` (needs specific MSR/PMU support and usually root;
attempted opportunistically, degrades gracefully to the Little's-Law
estimate when unavailable).

Output: a single JSON "resistance matrix" report per machine — this is
the artifact that later decides which of the other ideas in the analysis
are worth building (per §I.5).

**Caveat:** the current dev sandbox is a 4-core KVM VM with a virtio
block device — not representative of a real target box (the self-hosted
Linux server this is ultimately meant to run on). Numbers taken here are
a smoke test of the tool, not a real roofline. Re-run on the actual
target machine before trusting the results.

## Step 2 — Pick and test one falsifiable claim

Candidate from the analysis (§V.4):

> "Compute-aware weight reordering + credit-based prefetch improves
> tokens/s on a memory-constrained model by >20%."

Requires: a real model + llama.cpp (or similar) as the baseline, run on
real target hardware, with the Step 1 profiler's resistance matrix used
to decide the reordering. Not started — blocked on Step 1 running on
real hardware.

## Deliberately deferred / rejected (see analysis Part I.4 for why)

- Custom compiler / IR (§19, §36) — 20-person-year territory, not this
  project's job. Sit on top of ggml/llama.cpp, don't replace it.
- Cross-core *tile* passing (§14) — cache-line ping-pong trap. (Cross-core
  *activation* passing, §IV.1 of the analysis, is a different and later
  idea — model sharded resident-in-L2 across cores — not started.)
- Processing-near-storage (§23) — no real consumer hardware. Archived.
- Generic microkernel work (§9-§12) — already well-served by BLIS/ggml;
  low differentiation, not a priority.

## Longer-horizon ideas (not started, tracked for later)

From Part IV of the analysis, roughly in order of how self-contained
they are to prototype once Step 1/2 exist:

1. IV.3 — non-blocking river (never stall on a missing tile; degrade
   quality instead of throughput). Smallest to prototype, biggest
   philosophical shift.
2. IV.4 — compute-aware physical weight layout on disk (the "§F" idea
   from Part I.5 — cheap, measurable, nobody does this seriously).
3. IV.1 — model resident in L2/L3 per core, tokens travel instead of
   weights. Only viable for small (~1-2B, ternary/heavily quantized)
   models given current L3 sizes.
4. IV.5 — wide speculative decoding as an arithmetic-intensity
   transformer, not a latency trick.
5. IV.2 — generate weights from a seed+residual instead of transporting
   them. Most speculative, most compute-for-bytes trade.
6. IV.6 — single eternal kernel loop (no dispatch). Optimization on top
   of the others, not a starting point.
