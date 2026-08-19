# PLAN

Source documents: [continuous-data-river-original.md](continuous-data-river-original.md)
(the 39-section concept) and
[continuous-data-river-critical-analysis.md](continuous-data-river-critical-analysis.md)
(the critique + extensions — source of truth for what's actually in scope).
This file is the actionable distillation — update it as steps complete.

## Target hardware

Development happens in whatever sandbox is available; the intended real
target is a Docker container on a Hetzner server. The host is up and
reachable (see CLAUDE.md for the SSH command). The `tools/roofline`
image (see its README's "Run in Docker" section) now builds and runs
there — isolated from the host's other stacks (mercury/glitchtip/
supabase): its own `~/llmriver` directory, `llmriver-roofline` image/
container name (no collision with existing names), `--network none`,
`--rm` (one-shot, not a persistent service), bind mounts only for
report output and the disk-bench file. Host CPU is an AMD Ryzen 5 3600
(6c/12t, no AVX-512/VNNI — `compute` bench degrades to whatever
`cpu_info.cpp` detects); THP is in `madvise` mode, which is what
`hugepage` bench needs. Every benchmark number produced anywhere else
is a smoke test of the tooling, not a result to design around.

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

**Caveat:** the dev sandbox used earlier in this project is a 4-core KVM
VM with a virtio block device — not representative of a real target box,
and its numbers were a smoke test of the tool, not a real roofline.

**First real run:** done, on the Hetzner host (`mercury-hetzner`, AMD
Ryzen 5 3600, 6c/12t, 62GB RAM), via the isolated Docker setup described
in `tools/roofline/README.md`. Report saved at
[docs/reports/hetzner-mercury-2026-08-19.json](reports/hetzner-mercury-2026-08-19.json).
Notable numbers: single-thread RAM read 26.6 GB/s, aggregate read
saturates at ~44 GB/s by 3 threads (matches §III.1's "LFB wall, not
core-count wall" prediction); write-nt vs write-plain gap (25.5 vs
19.2 GB/s aggregate) confirms the RFO tax (§III.3.2); hugepage bench
confirmed 2MB backing with a 14.9% latency reduction (97.3ns → 82.8ns);
this CPU has no AVX-512/VNNI, so `compute` reported zero for both — the
INT4/INT8 GEMM proxy will need a non-AVX-512 path (AVX2, which this CPU
does have) before it means anything on this box. This was a 512MB/256MB
smoke-sized run (`--ram-buf-mb=256 --disk-size-mb=512`); re-run with
larger buffers for numbers to actually design around.

## Step 2 — Pick and test one falsifiable claim

Candidate from the analysis (§V.4):

> "Compute-aware weight reordering + credit-based prefetch improves
> tokens/s on a memory-constrained model by >20%."

Requires: a real model + llama.cpp (or similar) as the baseline, run on
real target hardware, with the Step 1 profiler's resistance matrix used
to decide the reordering.

**Baseline established** (`tools/llamacpp/`, vanilla upstream llama.cpp
`b10499`, unmodified — no llmRiver code involved, this is deliberately
the "before" number): Qwen2.5-7B-Instruct Q4_K_M, downloaded from the
official `Qwen/Qwen2.5-7B-Instruct-GGUF` repo (sha256-verified), running
on `mercury-hetzner` via the same isolated-Docker pattern as Step 1.

- Naive defaults (`-t 12`, all logical threads): pp512 41.2 tok/s,
  tg128 8.63 tok/s. Report:
  [docs/reports/llamacpp-baseline-qwen2.5-7b-q4km-2026-08-19.json](reports/llamacpp-baseline-qwen2.5-7b-q4km-2026-08-19.json).
- Tuned (`-t 6 -fa 1` — 6 = physical core count, flash attention on):
  9.15 tok/s decode, a ~6% free win over the naive default from flags
  alone, no algorithmic change. Sweep:
  [docs/reports/llamacpp-tuning-sweep-qwen2.5-7b-q4km-2026-08-19.json](reports/llamacpp-tuning-sweep-qwen2.5-7b-q4km-2026-08-19.json).
  Full sweep across 4/6/8/12 threads showed monotonically *decreasing*
  decode speed past 6 threads — direct empirical confirmation of
  §III.1's line-fill-buffer argument: SMT siblings share a physical
  core's LFBs rather than adding memory-level parallelism, so threads
  beyond the physical core count buy nothing and cost dispatch/sync
  overhead. See `tools/llamacpp/README.md` for the full table.
- This tuned number (9.15 tok/s) is ~97% of the ~9.4 tok/s ceiling
  implied by Step 1's measured aggregate RAM read bandwidth (~44 GB/s)
  divided by this model's size (4.68 GB) — i.e. vanilla llama.cpp is
  already close to the physical wall on this box. There is little room
  left for flag-tuning; beating this baseline by the claimed >20%
  requires the actual weight-reordering/prefetch work, not more knobs.

**Attempt 1 — killed.** `tools/llamacpp/patches/0001-madvise-hugepage-model-mmap.patch`:
`madvise(addr, size, MADV_HUGEPAGE)` on the model's mmap right after it's
created in `llama_mmap::impl` (`src/llama-mmap.cpp`), reasoning directly
from Step 1's own data — the roofline hugepage bench measured a 14.9%
latency win from 2MB pages on this exact host, and the generic mmap path
never requested them. Compute-aware in the sense the project means it:
grounded in this machine's own measured resistance matrix, not a guess.

Built (`llmriver-llamacpp:b10499-hugepage`), and *before* trusting any
benchmark number, checked whether the hint actually took effect the same
way Step 1's own tool does — via `/proc/<pid>/smaps` on the running
container. Result: the VMA got the `hg` flag (kernel acknowledged the
hint) but `THPeligible: 0` and `KernelPageSize: 4 kB` — the kernel never
actually backed it with a 2MB page. Root cause: this is a `MAP_SHARED`
read-only **file-backed** mapping over ext4; `MADV_HUGEPAGE`-driven THP
applies to anonymous memory (which is what Step 1's own latency-chase
buffer was) and to shmem/tmpfs, not to a plain ext4 file mmap on this
kernel (6.8). Benchmark result confirmed this exactly:
[docs/reports/llamacpp-hugepage-patch-qwen2.5-7b-q4km-2026-08-19.json](reports/llamacpp-hugepage-patch-qwen2.5-7b-q4km-2026-08-19.json) —
9.14 tok/s vs. the 9.15 tok/s tuned baseline, a 0.08% difference, inside
noise. Not "no improvement despite the fix working" — the fix never
activated, so this measures the null case exactly as predicted once the
smaps check was done. A clean kill, and cheap: found via a `/proc`
read, not by trusting the benchmark number in isolation.

What it would take to actually test this mechanism: get the weight file
onto a huge-page-capable backing (a `hugetlbfs` mount with pre-reserved
pages and `MAP_HUGETLB`, or a tmpfs copy, which support THP-style large
pages for file-backed mappings) instead of the plain ext4 mmap — a much
larger change than one `madvise` call, and not attempted yet.

**Attempt 2 — confirmed, +21.6%.** Not reordering/prefetch — a
different, complementary lever the ~3% ceiling math pointed straight at:
reduce physical bytes moved instead of trying to hide latency around a
fixed byte count (§I.3/§II.2's "physical resistance only bypassed by
moving fewer bytes"). Downloaded the same model at Q3_K_M instead of
Q4_K_M (3.80GB vs 4.68GB, ~19% fewer bytes/param, official
`Qwen/Qwen2.5-7B-Instruct-GGUF` repo, sha256-verified). Same tuned flags
(`-t 6 -fa 1`), clean run (no concurrent jobs this time — the first
attempt at this had a contention outlier from an overlapping job, redone
cleanly): **11.13 tok/s, stddev 0.003** — tight and reproducible.
Report: [docs/reports/llamacpp-q3km-qwen2.5-7b-2026-08-19.json](reports/llamacpp-q3km-qwen2.5-7b-2026-08-19.json).
21.6% over the 9.15 tok/s Q4_K_M baseline — clears the >20% bar, at the
honest cost this isn't free: Q3_K_M is a real quality/perplexity
downgrade from Q4_K_M, not just a config flag. Speed bought with
accuracy, not engineering.

**Attempt 3 — confirmed, but content-dependent (+10% to +45%).**
Speculative decoding: a small draft model (Qwen2.5-0.5B-Instruct Q8_0)
proposes tokens, the 7B target verifies several per forward pass. Same
weight-read cost amortized across multiple output tokens instead of
one — this is §IV.5/§II.5's idea ("speculate wide because width is
free") exactly, via llama.cpp's stock `--spec-draft-model` support, no
llmRiver code. Full writeup:
[docs/reports/llamacpp-speculative-decoding-qwen2.5-7b-2026-08-19.md](reports/llamacpp-speculative-decoding-qwen2.5-7b-2026-08-19.md).
On a predictable/technical prompt: 9.0 -> 13.1 tok/s (~45%, reproduced
across 3 runs). Verbose log showed 54% draft-token acceptance (draft
depth 3), ~2.6 output tokens per forward pass. On a creative/
unpredictable prompt: 9.0 -> 9.9 tok/s (~10%) — well under the bar.
**The gain is not a fixed multiplier; it tracks how well the tiny draft
model predicts the specific content being generated.** Any claim built
on this needs a range, not a single number.

Net: the original narrow candidate (reordering + credit-based prefetch)
is still untested as literally stated and its one attempt (huge pages)
was killed — but the broader Step 2 goal, beat this baseline by >20%
using an idea this project's own analysis actually predicts, has now
been hit twice, by two different mechanisms, each with a disclosed cost
(quantization: accuracy; speculative decoding: inconsistent, content-
dependent gain, plus the extra complexity/memory of running two
models). Neither is "free" the way the killed hugepage attempt would
have been if it had worked — matching §V.1's own framing: bytes are
expensive, and these two both pay for speed with something real.

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
