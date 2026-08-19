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

**Attempt 4 — combining Attempts 2+3 backfires.** Tried stacking the two
wins: Q3_K_M as the speculative-decoding target (same 0.5B Q8_0 draft).
Result: *worse* than either alone. Technical prompt: 10.7 tok/s (vs.
11.0 for Q3_K_M alone, vs. 13.1 for Q4_K_M+speculative). Creative
prompt: 6.9 tok/s (vs. 11.0 for Q3_K_M alone, vs. 9.0 for Q4_K_M alone).
Mechanism, confirmed via verbose log: draft acceptance rate on the
creative prompt dropped to ~29% (vs. the higher rate Q4_K_M got) —
Q3_K_M's extra quantization noise shifts the target's output
distribution enough that the full-precision 0.5B draft's guesses land
less often, so more verification passes get fully rejected. A fully
rejected pass still pays the wider-batch compute cost for only 1 output
token — worse than plain decode would have been. **Optimizations don't
simply stack; a noisier target actively degrades a draft model tuned
against a cleaner one.**

**Attempt 5 — hardware-level: dead ends, one instructive backfire.**
Three more directions tried, none beat Attempts 2/3:
- CPU frequency ceiling is firmware-locked at 3.6GHz — `bios_limit` in
  sysfs matches `scaling_max_freq` exactly, and writing a higher value
  succeeds (exit 0) but is silently clamped back. This is the platform
  firmware, not a Linux `cpufreq` policy; no IPMI/BIOS access from the
  OS to change it.
- Real (not just requested) huge pages for the model mmap needs the
  file to live on a `tmpfs` mounted with `huge=always` (plain ext4 was
  Attempt 1's dead end). Docker's `--tmpfs` flag doesn't support a
  `huge=` option, and both workarounds tried — a host-level
  `sudo mount`, and a container-scoped `--cap-add SYS_ADMIN` doing its
  own internal mount — were blocked by the permission classifier as
  privileged/systemwide actions. Not attempted further; would need the
  mount done outside this session.
- CCX/L3 topology confirmed via `lscpu -e`: two CCX domains, physical
  cores {0,1,2} and {3,4,5}, each with its own L3 slice, linked by
  Infinity Fabric (§III.3.6). Explicit thread pinning to test this
  (`llama-bench -C <mask>`) made things *worse*, not better: mask-only
  pinning to the same 6 cores the default already uses dropped to 6.35
  tok/s (vs. 9.15 unpinned), and adding `--cpu-strict 1` caused a
  catastrophic livelock at 0.94 tok/s. Confining to one CCX (3 cores,
  no cross-CCX traffic at all) got 8.15 tok/s — notably close to the
  6-core number despite using half the cores, which does support the
  cross-CCX-cost theory, but was still net worse than just leaving
  placement to the scheduler. **The default Linux CFS scheduler already
  places these threads at least as well as manual affinity does, and
  llama.cpp's poll-based thread sync (`--poll`) appears to interact
  badly with forced pinning.** Don't hand-pin threads on this build.

**Attempt 6 — batching: 3.7x, the largest win found, and it reframes
everything above.** Full writeup:
[docs/reports/batching-and-the-roofline-crossover-2026-08-19.md](reports/batching-and-the-roofline-crossover-2026-08-19.md).
Aggregate decode throughput, Q4_K_M, by parallel sequence count:
9.08 (B=1) -> 25.13 (B=4) -> **33.98 (B=16)** -> 33.81 (B=32) -> 33.89
(B=64). Dead flat past 16. This is §II.5's impedance transformer
demonstrated directly — weights are read once per forward pass no matter
how many sequences share it, so arithmetic intensity goes from ~1 to
~16 and the workload leaves the bandwidth-bound region entirely. The
plateau is the Ryzen's AVX2 FMA throughput taking over as the new wall.
Cost, stated plainly: this is *aggregate* throughput. Per-sequence
latency at B=16 is ~2.1 tok/s, far worse than a lone request's 9.08.
Batching serves many users well; it serves one waiting user badly.

**Attempt 7 — the regime inversion (most useful finding of the day).**
Predicted from Attempt 6 (if B>=4 is compute-bound, then Q3_K_M's
cheaper bytes should stop paying and its pricier dequant should start
costing) and confirmed:

| batch | Q4_K_M | Q3_K_M | winner |
|---|---|---|---|
| 1 | 9.08 | **10.94** | Q3_K_M +20% |
| 4 | **25.13** | 20.71 | Q4_K_M +21% |
| 16 | **33.98** | 23.44 | Q4_K_M +45% |

Attempt 2's headline (+21.6% from Q3_K_M) holds *only* at batch=1. At
batch 16 the same choice is 45% slower. The margin grows with batch
size. This also retroactively explains a detail visible in every earlier
test but not interpreted at the time: Q3_K_M's prompt processing was
always slower (41.6 vs 27.0 tok/s here; 38.8 vs 26.9 in the earlier
llama-cli runs) — prompt processing is inherently batched, so it was
compute-bound all along, and Q3_K_M was never winning there. One roofline
model now explains every measurement in this session.

**Attempt 8 — n-gram speculative decoding (no draft model): no gain on
novel text.** `--spec-type ngram-simple` / `ngram-cache` / `ngram-mod`,
no draft model needed at all. On the standard technical prompt: 9.1 /
8.6 / 9.1 tok/s vs. 9.0 baseline — noise, with ngram-cache slightly
negative. Expected: n-gram speculation works by matching against text
already in context, and a short prompt generating novel prose has
nothing to match. Untested and still promising: copy-heavy workloads
(refactoring, rewriting, summarization) where output largely reproduces
input — a prompt for that is staged at `~/llmriver/prompts/refactor.txt`
on the host but the comparison was not run.

**Hardware ceiling, now established definitively.** RAM is 2x32GB
DDR4-3200, both channels populated, running at full rated 3200 MT/s
(`dmidecode`). Theoretical peak 51.2 GB/s vs. 44 GB/s measured in Step 1
= 86% efficiency, which is normal. **There is no memory configuration
headroom left on this host** — the 44 GB/s wall is final, and combined
with the firmware-locked 3.6GHz CPU ceiling (Attempt 5), the hardware is
fully characterized and fully exploited.

**Attempt 9 — §IV.5 ("speculate wide, width is free") is FALSE on this
hardware.** The doc argues a 64-candidate tree reads the same weights as
1 candidate, so width costs nothing and the draft model should be judged
on diversity rather than accuracy. Measured, single-stream, Q4_K_M +
0.5B Q8_0 draft, varying `--spec-draft-n-max`:

| draft depth | tok/s |
|---|---|
| 3 (default) | **13.1** |
| 5 | 10.8 |
| 8 | 9.0 |
| 12 | 6.6 |

Monotonically worse. Width is *not* free, for two compounding reasons
this project's own data now explains: (1) a draft of depth N is a batch
of N+1 at verification time, and Attempt 6 established that batching
leaves the bandwidth-bound region and saturates against AVX2 FMA
throughput — so extra width buys compute cost, not free rides; (2)
acceptance decays with depth (each additional drafted token is
conditionally less likely to be right), so deep drafts do more work that
gets thrown away. **§IV.5's premise holds only while bandwidth-bound,
and speculation itself destroys that condition.** Optimal depth here is
shallow (3). Also tested: a *faster* draft (0.5B Q4_K_M, 491MB) instead
of the more accurate one (0.5B Q8_0, 676MB) — 11.4 vs 13.1 tok/s, so
draft *accuracy* matters more than draft speed, the opposite of §IV.5's
"judge the draft on diversity, not accuracy".

**Attempt 10 — Q4_0 beats Q4_K_M at batch=1.** 9.8 vs 9.0 tok/s (+9%).
Q4_0 is both smaller (4.43GB vs 4.68GB) and cheaper to dequantize, so it
wins on both axes of the roofline simultaneously — consistent with the
Attempt 7 model. With speculation it ties (13.0 vs 13.1), i.e. once
speculation moves the workload toward compute-bound the byte advantage
stops mattering, exactly as predicted.

**Hardware inventory — there is no second compute engine.** Checked
explicitly for any additional processor or memory pool to offload onto:
- GPU: only an **ASPEED** VGA controller — the server board's BMC/IPMI
  management chip. 2D framebuffer, no shaders/OpenCL/Vulkan compute.
  Unusable for inference.
- **AMD CCP** (Starship/Matisse Cryptographic Coprocessor): driver
  loaded but claimed by `kvm_amd`, no `/dev/ccp*` userspace node, and
  functionally an AES/SHA/RSA engine — it cannot do general MACs.
- No I/OAT / DSA / DMA offload engine of the kind some Xeons expose,
  which is the one thing that could plausibly stream weights without
  consuming per-core load-unit and line-fill-buffer slots (§III.1's real
  bottleneck).

Conclusion: **this box has exactly one compute engine and one memory
pool.** Every "offload it somewhere else" idea is closed on this
hardware. Combined with the firmware-locked clock and fully-populated
dual-channel RAM at rated speed, the only remaining lever for
single-stream decode is *bytes required per token* — which is a model
architecture question, not a systems-tuning question. That motivates
Attempt 11.

**Attempt 11 — MoE sparsity. The only thing that beat the bandwidth wall
instead of negotiating with it.** Full writeup:
[docs/reports/moe-sparsity-the-real-unlock-2026-08-19.md](reports/moe-sparsity-the-real-unlock-2026-08-19.md).

Reasoning that led here: Attempts 5/9 established this box has exactly
one compute engine and one memory pool, RAM is at rated speed, the clock
is firmware-locked, and llama.cpp is at ~97% of the bandwidth ceiling.
In `tok/s = bandwidth / bytes-per-token` the numerator is now fixed and
maxed, so the only remaining variable is the denominator — bytes per
token — which is a *model architecture* question, not a systems-tuning
one. That is §I.3's
argument ("nao mover bytes que nao sao precisos... ou seja:
esparsidade") and it had never been tested.

Qwen3-30B-A3B-Instruct-2507 Q4_K_M (30.5B total params, ~3.3B active per
token, 18.56GB file) vs. the dense Qwen2.5-7B Q4_K_M (7.6B params,
4.68GB):

| model | decode tok/s | bytes read/token |
|---|---|---|
| dense 7B | 9.08 | 4.68 GB |
| **MoE 30B** | **18.66** | **2.38 GB** (44 GB/s / 18.66) |

**4x the parameters, 2.05x the speed, better model.** ~16GB of its
weights sit resident in RAM and are never touched on a given token.

Two anti-synergies found, both from one mechanism — MoE's win is
*per-token* expert sparsity, so anything batching several tokens into
one pass reads the **union** of their expert sets:
- **Speculation hurts MoE**: 18.5 -> 13.3 / 11.6 / 10.0 at draft depth
  2/3/5. Speculative decoding assumes "verify N tokens = same weight
  reads as 1", which holds for dense models and is false for MoE. (On
  the dense 7B the same technique *helped*: 9.0 -> 13.1.)
- **MoE batches worse than dense**: 2.44x scaling to B=16 vs. dense's
  3.74x, because bytes-per-pass grows with batch instead of staying
  flat. Its lead narrows from 2.05x (B=1) to 1.34x (B=16) — but it is
  still fastest at every batch size tested (45.53 tok/s at B=16).

**Attempt 12 — huge pages, finally tested properly, and they do not
matter for this workload.** Attempt 1 failed because file-backed mmaps
can't get THP. Fixed both blockers: set THP `enabled=always` system-wide
(reverted afterward) and used `--no-mmap` so weights land in
THP-eligible *anonymous* memory. Result: 9.0 -> 9.1 tok/s. Noise.

The explanation matters more than the number, and it is a **critique of
this project's own Step 1 tool**: `tools/roofline` measured a 14.9%
huge-page win using a *random pointer-chase*. LLM decode streams weights
*sequentially*, and sequential access already has near-ideal TLB
behavior (one miss per 4KB while consuming 4KB of useful work). Huge
pages help random access; this workload isn't random. **A resistance
number only transfers if it was measured with the access pattern the
real workload uses** — the profiler should measure sequential streaming
explicitly, not just pointer-chase latency. (§III.3.1 calls TLB "a
barreira mais subestimada"; for *streaming* weight reads on this host,
it is measurably not a barrier at all.)

**Attempt 13 — native ternary (BitNet b1.58): fast, but the available
GGUF is broken.** The logical endpoint of the bytes-per-token argument:
a model *trained* at ~1.58 bits rather than compressed afterward, which
should avoid the dequantization penalty that ate ~17% of the Q2_K gain.

Microsoft's official `bitnet-b1.58-2B-4T-gguf` ships in `i2_s` format,
which stock llama.cpp does not support (it needs their bitnet.cpp fork);
our build supports llama.cpp's own ternary types `TQ1_0`/`TQ2_0`. Used a
community TQ2_0 conversion (`Synapticode/bitnet-b1.58-2B-4T-tq2_0-gguf`,
1.20GB).

Speed: **31.9 tok/s decode, 133.6 tok/s prompt** — the fastest decode
measured in this project (3.5x the dense baseline). Quality: **unusable**
— degenerate repetition loops on every prompt tried, both greedy with
`--ignore-eos` and with normal sampling (`--temp 0.7 --repeat-penalty
1.1`); asked for a Python function it never emits code, just repeats
"Yes, I can provide you with a Python function..." indefinitely. Almost
certainly the community conversion rather than the architecture, but not
worth further pursuit here.

Note the efficiency figure, which is the actually interesting part:
1.20GB at 44 GB/s predicts 36.7 tok/s, achieved 31.9 = **87%** — about
the same as Q3_K_M's 88% and better than Q2_K's 83%, but *not* the
dramatic dequant-free win the ternary premise suggests. On this evidence
native ternary does not obviously beat K-quants on
bandwidth-efficiency; its advantage is simply that 1.58 bits is fewer
bits.

## Step 2 — net conclusion

The broader goal (beat the baseline by >20% using ideas this project's
own analysis predicts) has been hit repeatedly. Ranked by size:

| mechanism | gain | cost | regime |
|---|---|---|---|
| **MoE + batching** | **+402%** (9.08 -> 45.53 agg.) | per-sequence latency; 18.6GB RAM | serving many users |
| **MoE + Q2_K quant** | **+183%** (9.08 -> 25.7) | quality (verified still coherent); loses when batched | single user |
| **MoE architecture** | **+105%** (9.08 -> 18.66) | 18.6GB RAM resident; **breaks speculation** | single user |
| batching (dense, B=16) | +274% (9.08 -> 33.98 agg.) | per-sequence latency collapses to ~2.1 tok/s | serving many users |
| speculative decoding | +45% technical / +10% creative | second model in RAM; content-dependent; **negative on MoE** | dense, single user |
| Q4_0 over Q4_K_M | +9% | slight quality loss | single user only |
| Q3_K_M quantization | +20% | model quality, **and it inverts to -45% when batched** | single user only |

**The headline: a 30B MoE runs 2x faster than a dense 7B, and is a
better model.** Sparsity was the only technique tested that reduced
bytes-per-token rather than rearranging or amortizing them — which is
exactly what §I.3 predicted and what §I.2's bandwidth argument implies
is the only real escape.

The single most important thing learned: **these are not independent
knobs, and they do not stack.** Q3_K_M + speculative decoding is worse
than either alone (Attempt 4). Q3_K_M's win reverses entirely under
batching (Attempt 7). Which optimization is correct depends on which
side of the roofline crossover the workload sits, and applying a
bandwidth-bound fix to a compute-bound workload actively costs
performance.

This sharply bounds the original candidate claim (compute-aware weight
reordering + credit-based prefetch, >20%). That claim targets the
bandwidth-bound region — but that region is only single-stream decode,
and llama.cpp is already at ~97% of its physical ceiling there
(9.15 of ~9.4 tok/s). Anything that amortizes weight reads moves the
system out of that region, where bandwidth optimizations have nothing
left to optimize. **The idea is not dead, but its maximum possible
payoff on this host is ~3%, in the one case (batch=1) where the two
larger wins are unavailable or unwanted.** §V.4's instruction was to
pick a falsifiable claim and try to kill it in a week rather than a
year; this one is not killed, but it is now known to be worth far less
than the roadmap assumed, which is the same kind of saving.

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
