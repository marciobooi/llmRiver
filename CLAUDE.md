# llmriver — CPU Data-Movement Layer for LLM Inference

Research/engineering project: CPU-bound LLM inference is a data-movement
problem wearing a compute problem's clothes. This repo treats weight
loading, caching, and compute as a pipelined "river" of bytes with
measurable **resistance**, and builds the measurement/control layer that
decides which of the underlying ideas are worth building — instead of
building a runtime or compiler on faith.

**Status:** early. `tools/roofline/` (a per-machine resistance-matrix
profiler) exists and works. No inference runtime exists yet, and none
should until Step 1 in [docs/PLAN.md](docs/PLAN.md) has run on real target
hardware.

## Source documents — read before proposing new work

- [docs/continuous-data-river-original.md](docs/continuous-data-river-original.md)
  — the original 39-section concept (Portuguese). Ambitious, some of it
  right, a lot of it either already-solved-elsewhere or undeliverable at
  this project's scale.
- [docs/continuous-data-river-critical-analysis.md](docs/continuous-data-river-critical-analysis.md)
  — the critique of that concept and the actual scope of this project
  (Portuguese). **This is the source of truth for what's in scope, not
  the original doc.** In particular: Part I.4 lists what's explicitly cut
  (custom compiler/IR, cross-core tile passing, processing-near-storage,
  generic microkernel work already done by BLIS/ggml); Part I.5 says what's
  actually novel and buildable (the measurement/control layer, sitting on
  top of ggml/llama.cpp, not replacing it); Part V.4 is the concrete
  first step.
- [docs/PLAN.md](docs/PLAN.md) — the actionable, current-status distillation
  of the above two. Check this before re-deriving the roadmap from the
  source docs each session.

Both source documents are kept in Portuguese, verbatim, as the canonical
record of the original thinking. Do not translate them or paraphrase over
them in place — write new analysis in PLAN.md or new docs instead.

## The one rule that matters more than any other

**Measure before you build.** Do not write inference-runtime or compiler
code because it seems like the obvious next step. The critical-analysis
doc's own conclusion (§V.4) is: build the roofline of one specific
machine, then pick one falsifiable claim and try to kill it. If you're
about to add a feature and can't point to which resistance-matrix number
motivates it, stop and ask instead of proceeding.

Corollary: §19/§36 territory (a custom LLM compiler/IR, TVM/IREE/
tinygrad-scale work) is explicitly out of scope per the analysis — it's
flagged there as the point where a document this ambitious quietly
becomes undeliverable. Don't drift into it.

## Target hardware

The eventual real target is a Docker container running on a Hetzner
server. The host is reachable now:

```
ssh -i ~/.ssh/mercury_admin mercury@65.109.96.74
```

The `tools/roofline` Docker image now builds and runs on that host,
isolated from its other stacks (mercury/glitchtip/supabase) — see
`tools/roofline/README.md`'s "Run in Docker" section for the exact
commands and docs/PLAN.md's "Target hardware" / Step 1 sections for the
first real report. Any code that hardcodes assumptions from a dev
sandbox's hardware (core count, cache sizes, NUMA topology) is still a
bug — the host's own numbers (AMD Ryzen 5 3600, no AVX-512) are just as
easy to over-fit to as a VM's.

## Stack / working conventions

- `tools/roofline/`: C++17, CMake, zero external dependencies beyond
  pthreads. Built with `-march=native` — it's meant to be rebuilt on the
  machine being profiled, never distributed as a portable binary.
- No dependency on a specific vendor ISA (AVX-512/AMX/CUDA/etc.) without a
  runtime capability check and a graceful degrade path — this project
  inherits the original doc's "hardware agnostic" principle (§15) even
  though most of the doc's grander architecture around it is out of
  scope. See `cpu_info.cpp` / the `#if defined(__AVX512F__)` guards in
  `compute_bench.cpp` for the pattern.
- Real error handling at system boundaries (file I/O, mmap, hardware
  feature detection); no silent fallback that hides a measurement being
  wrong. Degraded modes (no O_DIRECT, no confirmed hugepages, no VNNI)
  must show up in the report/notes, not just in stderr.
- No comments explaining *what* code does. A comment earns its place only
  when it records *why* — a hidden constraint, a non-obvious hardware
  fact, a citation back to the specific section of the analysis doc that
  motivates a design choice (the existing code does this throughout;
  match that style).
- Keep this file short. Put implementation detail in `tools/roofline/README.md`
  and reasoning in `docs/*.md`.
- End any session that changes the repo by updating
  [docs/PLAN.md](docs/PLAN.md)'s status/checkboxes so the next session
  doesn't have to re-derive what's done.
