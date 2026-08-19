#pragma once

namespace roofline {

struct ComputeBenchResult {
  bool fp32_fma_available = false;
  double fp32_fma_gflops = 0.0;

  bool int8_vnni_available = false;
  double int8_vnni_gops = 0.0;

  // Both kernels loop over a buffer sized to stay resident in L1 so the
  // measurement reflects execution-port/FMA throughput, not memory
  // bandwidth -- this is the "GEMM throughput achievable" number from
  // §V.4, not a memory-bound end-to-end number.
};

ComputeBenchResult run_compute_bench(int duration_ms);

}  // namespace roofline
