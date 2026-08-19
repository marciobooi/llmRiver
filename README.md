# llmriver

CPU-bound LLM inference is a data-movement problem wearing a compute
problem's clothes. `llmriver` is a research/engineering project about
that reframe: treating weight loading, caching, and compute as a
pipelined river of bytes with measurable **resistance**, rather than a
stack of independent stages tuned in isolation.

This repo starts from a critical analysis of the original "Continuous
Data River" concept — see
[docs/continuous-data-river-critical-analysis.md](docs/continuous-data-river-critical-analysis.md)
for the full write-up (Portuguese). Short version:

- Pipelining hides **latency**. It does nothing for **bandwidth**. A
  40 GB model over a 7 GB/s NVMe link is ~0.17 tok/s no matter how
  good the pipeline is — the river gets you to the wall faster, it
  doesn't move the wall.
- The real unlock is **not moving bytes that aren't needed**
  (sparsity: MoE expert routing, FFN neuron sparsity) — and once
  loading is sparse, it's also unpredictable, which is exactly where a
  real river architecture (speculative loads, backpressure, misprediction
  discard) stops being optional and starts being required.
- What's actually novel and buildable, on top of existing kernels
  (llama.cpp/ggml, BLIS-style microkernels) rather than instead of
  them, is the **measurement and control layer**: a resistance matrix
  of the real machine, a profiler that finds the actual bottleneck, an
  auto-tuner, and a feedback loop. Nobody has built this.

See [docs/PLAN.md](docs/PLAN.md) for the actionable roadmap and current
status.

## Status

Early. First deliverable is the machine **roofline / resistance-matrix
profiler** (`tools/roofline/`) — measuring the real physical resistances
(RAM bandwidth, disk sequential/random throughput, achievable compute
throughput, TLB/hugepage effects, implied line-fill-buffer concurrency)
on the target machine, per the project's own rule: measure before
building any runtime, then pick one falsifiable claim and try to kill
it. No inference runtime exists yet.

## Layout

```
docs/
  continuous-data-river-critical-analysis.md   the design doc (source of truth)
  PLAN.md                                       roadmap / current status
tools/
  roofline/                                     machine resistance-matrix profiler (C++)
```

## Building

```
cd tools/roofline
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/roofline all
```

See [tools/roofline/README.md](tools/roofline/README.md) for details,
flags, and what the JSON report means.
