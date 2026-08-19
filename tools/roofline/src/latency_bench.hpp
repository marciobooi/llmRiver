#pragma once

#include <cstddef>
#include <vector>

namespace roofline {

struct MlpPoint {
  int streams;
  double bandwidth_gbs;
};

struct LatencyBenchResult {
  size_t footprint_bytes = 0;

  // Average latency of a single dependent pointer-chase hop (each load's
  // address depends on the previous load's value, so the core cannot have
  // more than one of these outstanding at a time). This is the closest a
  // userspace benchmark gets to "real" random DRAM access latency.
  double dependent_chain_latency_ns = 0.0;

  // Achieved random-access bandwidth as a function of how many independent
  // pointer-chase streams are interleaved in the same thread (i.e. how much
  // memory-level parallelism software exposes to the core). Bandwidth
  // should grow roughly linearly with stream count until something caps
  // it -- per §III.1 of the analysis, that cap is expected to be the core's
  // line-fill-buffer count, not the DRAM controller.
  std::vector<MlpPoint> mlp_scaling;
  int mlp_plateau_streams = 0;
  double mlp_plateau_bandwidth_gbs = 0.0;

  // Little's Law estimate: bytes_in_flight = bandwidth * latency, divided
  // by a 64B cache line, evaluated at the plateau. This is the empirical
  // proxy for "how many concurrent outstanding misses is this core really
  // sustaining" -- compare against the ~10-16 line-fill-buffers-per-core
  // figure cited in the analysis. It is a proxy, not a PMU measurement:
  // treat it as an order-of-magnitude cross-check, not a hardware count.
  double implied_outstanding_requests = 0.0;
};

// footprint_bytes should comfortably exceed L3 size so hops are real DRAM
// (or at least LLC-miss) accesses, not L2/L3 hits.
LatencyBenchResult run_latency_bench(size_t footprint_bytes, int max_streams,
                                      int duration_ms);

}  // namespace roofline
