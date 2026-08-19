# roofline

A machine resistance-matrix profiler. Measures the real, physical
resistances a CPU-bound LLM inference runtime would fight on *this*
machine, instead of assuming them from spec sheets. See
[../../docs/PLAN.md](../../docs/PLAN.md) Step 1 and
[../../docs/continuous-data-river-critical-analysis.md](../../docs/continuous-data-river-critical-analysis.md)
§V.4 for why this is the first thing to build, before any runtime code.

No external dependencies beyond a C++17 compiler and pthreads.

## Build

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Built with `-march=native`: this is meant to be compiled *on* the machine
you're profiling, not distributed as a portable binary. Rebuild it on the
target box.

## Run

```
./build/roofline all                       # everything, human-readable
./build/roofline all --json --out=report.json
./build/roofline ram --ram-buf-mb=1024      # just the RAM sweep
./build/roofline disk --disk-file=/mnt/nvme/test.bin --disk-size-mb=8192
```

Subcommands: `all`, `ram`, `latency`, `disk`, `compute`, `hugepage`.
Run with no arguments for the full flag list.

## What each part measures, and why

- **ram** — sequential streaming read/write/write-non-temporal bandwidth,
  swept across thread counts. The single-thread number is expected to sit
  near a per-core ceiling set by line-fill-buffer count, not near full DRAM
  bandwidth (analysis §III.1); the point where adding threads stops
  helping is the empirical "how many cores does it take to generate enough
  memory-level parallelism" answer. The write-plain-vs-write-non-temporal
  gap is the RFO tax from §III.3.2.
- **latency** — dependent pointer-chase latency (real random-access
  latency, Sattolo single-cycle so no prefetcher/predictor can shortcut
  it), plus a sweep of independent in-flight streams (§III.1's
  "memory-level parallelism") whose bandwidth plateau gives, via Little's
  Law, an empirical proxy for how many outstanding misses this core is
  really sustaining — the thing actual line-fill-buffer counts would tell
  you directly if this tool could read the PMU (it can't, portably,
  without root; this is the honest substitute).
- **disk** — sequential and random-4K throughput against a real file,
  O_DIRECT where the filesystem allows it. This is "how fast can weights
  actually stream off storage," the number §I.2 of the analysis uses to
  show pipelining can't fix a bandwidth-bound wall.
- **compute** — AVX-512 FP32 FMA and VNNI INT8 throughput, L1-resident so
  it reflects execution-port/FMA throughput rather than memory bandwidth.
  Stands in for "achievable GEMM throughput" until a specific INT4/INT8
  weight-unpack kernel exists.
- **hugepage** — the same pointer-chase latency test run over a
  4K-page-pinned buffer vs. a `MADV_HUGEPAGE` buffer, to directly quantify
  the TLB/page-walk cost (analysis §III.3.1's "the most underestimated
  barrier"). Confirms via `/proc/self/smaps` whether the kernel actually
  backed the mapping with 2MB pages, since `MADV_HUGEPAGE` is only a hint.

## Known limitations

- No PMU/line-fill-buffer occupancy counters — that needs
  `perf_event_open` with specific hardware/kernel/permission support this
  tool doesn't assume. The MLP-sweep-plateau in `latency` is the
  deliberate substitute.
- `l3_bytes_total` in `cpu` reads a single sysfs cache instance (cpu0's).
  On a monolithic single-instance L3 (typical Intel ring/mesh) that's the
  true machine total; on multi-instance topologies (e.g. AMD multi-CCX)
  it under-reports.
- The `disk` bench falls back to buffered reads with
  `posix_fadvise(DONTNEED)` when O_DIRECT isn't supported by the
  filesystem (common on overlayfs/tmpfs container roots) — the report's
  `used_o_direct` / `note` fields say when this happened. Point
  `--disk-file` at a real block-device-backed path for trustworthy
  numbers.
- No llama.cpp stall-breakdown integration yet (see docs/PLAN.md).
