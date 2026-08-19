#pragma once

#include <cstddef>
#include <string>

namespace roofline {

struct HugepageBenchResult {
  bool ran = false;
  std::string note;

  size_t footprint_bytes = 0;

  double baseline_4k_latency_ns = 0.0;   // regular 4K pages, THP disabled
  double hugepage_latency_ns = 0.0;      // THP-backed, best effort
  bool hugepage_confirmed = false;       // verified via /proc/self/smaps
  double latency_reduction_pct = 0.0;    // positive = hugepages helped
};

// Random-access pointer-chase latency over a buffer explicitly kept on 4K
// pages vs. the same footprint with MADV_HUGEPAGE requested, to isolate
// the TLB-miss / page-walk cost §III.3.1 of the analysis calls the most
// underestimated barrier. footprint_bytes should be well beyond L3 size.
HugepageBenchResult run_hugepage_bench(size_t footprint_bytes, int duration_ms);

}  // namespace roofline
