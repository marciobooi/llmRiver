#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace roofline {

struct ThreadScalingPoint {
  int threads;
  double aggregate_gbs;
  double per_thread_gbs;
};

struct RamBenchResult {
  // Sequential streaming bandwidth vs. thread count, for three kernels.
  // read: buf[i] summed into an accumulator (pure load traffic).
  // write_plain: buf[i] = pattern (a plain store -> triggers RFO, so this
  //   measures ~2x the "useful" write bandwidth in traffic terms).
  // write_nt: buf[i] stored via a non-temporal (streaming) store, which
  //   bypasses RFO -- this is what III.3.2 in the analysis is about.
  std::vector<ThreadScalingPoint> read_scaling;
  std::vector<ThreadScalingPoint> write_plain_scaling;
  std::vector<ThreadScalingPoint> write_nt_scaling;

  size_t buffer_bytes_per_thread = 0;
  int max_threads_tested = 0;

  // Single-thread ceiling (threads==1 point of read_scaling), called out
  // separately because it's the number §III.1 says should sit near the
  // line-fill-buffer bound (~9.6 GB/s order of magnitude on a Skylake-class
  // core with ~12 LFBs and ~80ns DRAM latency), not near full DRAM
  // bandwidth.
  double single_thread_read_gbs = 0.0;

  // Thread count at which aggregate read bandwidth stopped growing by more
  // than 5% per added thread -- the empirical "this many cores are needed
  // just to generate enough memory-level parallelism to saturate DRAM"
  // point that §III.1 predicts should land around 4-6 cores.
  int read_saturation_threads = 0;
  double read_saturation_gbs = 0.0;
};

// buffer_bytes_per_thread should comfortably exceed the machine's L3 size
// (per thread) so the benchmark measures DRAM, not cache, bandwidth.
RamBenchResult run_ram_bench(size_t buffer_bytes_per_thread, int max_threads,
                              int duration_ms);

}  // namespace roofline
