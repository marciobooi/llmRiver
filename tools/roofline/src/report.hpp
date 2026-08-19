#pragma once

#include <optional>
#include <ostream>

#include "compute_bench.hpp"
#include "cpu_info.hpp"
#include "disk_bench.hpp"
#include "hugepage_bench.hpp"
#include "latency_bench.hpp"
#include "ram_bench.hpp"

namespace roofline {

struct Report {
  CpuInfo cpu;
  std::optional<RamBenchResult> ram;
  std::optional<LatencyBenchResult> latency;
  std::optional<DiskBenchResult> disk;
  std::optional<ComputeBenchResult> compute;
  std::optional<HugepageBenchResult> hugepage;
};

// The "resistance matrix": one JSON document per machine, meant to be
// diffed across machines/configs and fed into whatever picks the
// falsifiable claim in docs/PLAN.md Step 2.
void write_json(std::ostream& out, const Report& report);

// Human-readable summary with the interpretive notes tied to the analysis
// document (what each number is supposed to tell you).
void write_human(std::ostream& out, const Report& report);

}  // namespace roofline
