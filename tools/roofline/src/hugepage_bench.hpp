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

  double baseline_4k_seq_gbs = 0.0;      // sequential streaming, 4K pages
  double hugepage_seq_gbs = 0.0;         // sequential streaming, THP-backed
  double seq_gain_pct = 0.0;             // positive = hugepages helped
};

// Compares 4K pages against MADV_HUGEPAGE under BOTH access patterns, because
// the two give very different answers and only one of them predicts a given
// workload.
//
// Random (pointer-chase) isolates the TLB-miss / page-walk cost that §III.3.1
// calls the most underestimated barrier. Sequential streaming is what
// weight-reading during LLM decode actually does, and it was measured on
// mercury-hetzner to gain nothing from hugepages (see
// docs/reports/moe-sparsity-the-real-unlock-2026-08-19.md): one TLB miss per
// 4K page while consuming a full 4K of useful data is already negligible
// overhead, so there is little for a 2MB page to remove.
//
// Reporting only the random number is what previously produced a headline
// 14.9% "hugepage win" on this host that did not transfer to inference at all.
// footprint_bytes should be well beyond L3 size.
HugepageBenchResult run_hugepage_bench(size_t footprint_bytes, int duration_ms);

}  // namespace roofline
