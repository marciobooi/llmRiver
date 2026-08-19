#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace roofline {

struct DiskBenchResult {
  std::string file_path;
  uint64_t file_size_bytes = 0;
  bool used_o_direct = false;
  std::string note;  // set when something degraded (e.g. no O_DIRECT support)

  double sequential_read_gbs = 0.0;
  double random4k_iops = 0.0;
  double random4k_bandwidth_mbs = 0.0;
  double random4k_avg_latency_us = 0.0;
};

// Creates `path` filled with `file_size_bytes` (rounded up to a 4MiB
// boundary) if it doesn't already exist at that size, then measures
// sequential and random-4K read throughput against it. Prefers O_DIRECT to
// bypass the page cache; falls back to buffered reads with
// posix_fadvise(DONTNEED) if the filesystem rejects O_DIRECT (common on
// overlayfs/tmpfs container root filesystems), and records that fact in
// `note` / `used_o_direct`.
DiskBenchResult run_disk_bench(const std::string& path, uint64_t file_size_bytes,
                                int duration_ms);

}  // namespace roofline
