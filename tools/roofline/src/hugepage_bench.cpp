#include "hugepage_bench.hpp"

#include <sys/mman.h>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "chase.hpp"
#include "util.hpp"

namespace roofline {
namespace {

std::atomic<uint64_t> g_sink{0};

size_t round_up(size_t v, size_t mult) { return ((v + mult - 1) / mult) * mult; }

double measure_dependent_latency(const ChaseNode* arr, int duration_ms) {
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  uint32_t cur = 0;
  uint64_t hops = 0;
  uint64_t start = now_ns();
  do {
    for (int u = 0; u < 64; ++u) cur = arr[cur].next;
    hops += 64;
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;
  g_sink.fetch_add(cur, std::memory_order_relaxed);
  return static_cast<double>(elapsed) / static_cast<double>(hops);
}

// Touches one 8-byte word per 64-byte line so every cache line in the
// footprint is pulled in exactly once per pass, which is the access shape of
// streaming a weight tensor.
double measure_sequential_bandwidth(const void* mem, size_t size, int duration_ms) {
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  const auto* p = static_cast<const uint64_t*>(mem);
  size_t words = size / sizeof(uint64_t);
  uint64_t acc = 0;
  uint64_t bytes = 0;
  uint64_t start = now_ns();
  do {
    for (size_t i = 0; i < words; i += 8) acc += p[i];
    bytes += size;
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;
  g_sink.fetch_add(acc, std::memory_order_relaxed);
  if (elapsed == 0) return 0.0;
  return static_cast<double>(bytes) / static_cast<double>(elapsed);
}

struct Vma {
  uint64_t start;
  uint64_t end;
};

bool parse_header(const std::string& line, Vma& out) {
  size_t dash = line.find('-');
  if (dash == std::string::npos || dash == 0) return false;
  for (size_t i = 0; i < dash; ++i) {
    if (!std::isxdigit(static_cast<unsigned char>(line[i]))) return false;
  }
  size_t sp = line.find(' ', dash);
  if (sp == std::string::npos) return false;
  out.start = std::strtoull(line.substr(0, dash).c_str(), nullptr, 16);
  out.end = std::strtoull(line.substr(dash + 1, sp - dash - 1).c_str(), nullptr, 16);
  return true;
}

// Sums the "AnonHugePages" field of every /proc/self/smaps VMA whose start
// address falls inside [target_start, target_end). A non-zero sum means the
// kernel actually backed (some of) that mapping with 2MB pages, as opposed
// to MADV_HUGEPAGE being silently ignored.
uint64_t confirmed_hugepage_bytes(uintptr_t target_start, uintptr_t target_end) {
  std::ifstream f("/proc/self/smaps");
  if (!f) return 0;
  std::string line;
  bool in_range = false;
  uint64_t sum_bytes = 0;
  while (std::getline(f, line)) {
    Vma v;
    if (parse_header(line, v)) {
      in_range = v.start >= target_start && v.start < target_end;
      continue;
    }
    if (in_range && line.rfind("AnonHugePages:", 0) == 0) {
      uint64_t kb = 0;
      std::istringstream iss(line.substr(std::strlen("AnonHugePages:")));
      iss >> kb;
      sum_bytes += kb * 1024;
    }
  }
  return sum_bytes;
}

}  // namespace

HugepageBenchResult run_hugepage_bench(size_t footprint_bytes, int duration_ms) {
  HugepageBenchResult result;
  size_t size = round_up(footprint_bytes, 2 * 1024 * 1024);
  result.footprint_bytes = size;
  size_t count = size / sizeof(ChaseNode);

  void* base_mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  void* huge_mem = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base_mem == MAP_FAILED || huge_mem == MAP_FAILED) {
    result.note = "mmap failed, skipping hugepage comparison";
    if (base_mem != MAP_FAILED) munmap(base_mem, size);
    if (huge_mem != MAP_FAILED) munmap(huge_mem, size);
    return result;
  }

  madvise(base_mem, size, MADV_NOHUGEPAGE);
  madvise(huge_mem, size, MADV_HUGEPAGE);

  auto* base_arr = static_cast<ChaseNode*>(base_mem);
  auto* huge_arr = static_cast<ChaseNode*>(huge_mem);
  wire_single_cycle(base_arr, count, /*seed=*/0xb45e1111ULL);
  wire_single_cycle(huge_arr, count, /*seed=*/0xb45e2222ULL);

  result.baseline_4k_latency_ns = measure_dependent_latency(base_arr, duration_ms);
  result.hugepage_latency_ns = measure_dependent_latency(huge_arr, duration_ms);

  result.baseline_4k_seq_gbs = measure_sequential_bandwidth(base_mem, size, duration_ms);
  result.hugepage_seq_gbs = measure_sequential_bandwidth(huge_mem, size, duration_ms);
  if (result.baseline_4k_seq_gbs > 0) {
    result.seq_gain_pct = (result.hugepage_seq_gbs - result.baseline_4k_seq_gbs) /
                          result.baseline_4k_seq_gbs * 100.0;
  }

  uint64_t confirmed = confirmed_hugepage_bytes(
      reinterpret_cast<uintptr_t>(huge_mem), reinterpret_cast<uintptr_t>(huge_mem) + size);
  result.hugepage_confirmed = confirmed > 0;
  if (!result.hugepage_confirmed) {
    result.note =
        "MADV_HUGEPAGE was requested but /proc/self/smaps shows 0 "
        "AnonHugePages -- the kernel did not actually back this mapping "
        "with 2MB pages (common in small/fragmented-memory sandboxes). "
        "The 'hugepage' numbers below do not reflect a real hugepage "
        "benefit in that case.";
  }

  if (result.baseline_4k_latency_ns > 0) {
    result.latency_reduction_pct =
        (result.baseline_4k_latency_ns - result.hugepage_latency_ns) /
        result.baseline_4k_latency_ns * 100.0;
  }
  result.ran = true;

  munmap(base_mem, size);
  munmap(huge_mem, size);
  return result;
}

}  // namespace roofline
