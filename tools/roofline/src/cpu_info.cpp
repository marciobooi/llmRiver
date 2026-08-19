#include "cpu_info.hpp"

#include <unistd.h>

#include <cctype>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace roofline {

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string trim(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  return s.substr(a, b - a + 1);
}

// Parses sysfs cache size strings like "32K" or "1280K" into bytes.
uint64_t parse_cache_size(const std::string& raw) {
  std::string s = trim(raw);
  if (s.empty()) return 0;
  uint64_t mult = 1;
  char suffix = s.back();
  if (suffix == 'K' || suffix == 'k') {
    mult = 1024;
    s.pop_back();
  } else if (suffix == 'M' || suffix == 'm') {
    mult = 1024 * 1024;
    s.pop_back();
  }
  try {
    return static_cast<uint64_t>(std::stoull(s)) * mult;
  } catch (...) {
    return 0;
  }
}

std::string model_name_from_proc_cpuinfo() {
  std::ifstream f("/proc/cpuinfo");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("model name", 0) == 0) {
      auto pos = line.find(':');
      if (pos != std::string::npos) return trim(line.substr(pos + 1));
    }
  }
  return "unknown";
}

void detect_caches(CpuInfo& info) {
  const std::string base = "/sys/devices/system/cpu/cpu0/cache/";
  for (int idx = 0; idx < 8; ++idx) {
    std::string dir = base + "index" + std::to_string(idx) + "/";
    std::string level_s = trim(read_file(dir + "level"));
    std::string type_s = trim(read_file(dir + "type"));
    std::string size_s = trim(read_file(dir + "size"));
    if (level_s.empty()) continue;  // no such index
    uint64_t bytes = parse_cache_size(size_s);
    if (level_s == "1" && type_s == "Data") {
      info.l1d_bytes_per_core = bytes;
    } else if (level_s == "2") {
      info.l2_bytes_per_core = bytes;
    } else if (level_s == "3") {
      // sysfs reports the size of this L3 *instance*; on a monolithic
      // single-instance L3 (typical Intel ring/mesh) this is the true
      // total. On multi-instance topologies (e.g. AMD multi-CCX) this
      // under-reports the machine total — see docs/PLAN.md caveat.
      info.l3_bytes_total = bytes;
    }
  }
}

void detect_hugepages(CpuInfo& info) {
  std::string thp = read_file(
      "/sys/kernel/mm/transparent_hugepage/enabled");
  info.thp_available =
      thp.find("[always]") != std::string::npos ||
      thp.find("[madvise]") != std::string::npos;

  std::string free_s = trim(
      read_file("/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages"));
  if (!free_s.empty()) {
    try {
      info.hugetlb_available = std::stoull(free_s) > 0;
    } catch (...) {
      info.hugetlb_available = false;
    }
  }
}

void detect_isa(CpuInfo& info) {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_cpu_init();
  info.has_avx2 = __builtin_cpu_supports("avx2");
  info.has_avx512f = __builtin_cpu_supports("avx512f");
  info.has_avx512bw = __builtin_cpu_supports("avx512bw");
  info.has_avx512vnni = __builtin_cpu_supports("avx512vnni");
#if defined(__GNUC__) && __GNUC__ >= 10 || defined(__clang__)
  info.has_avx512_bf16 = __builtin_cpu_supports("avx512bf16");
#endif
#endif
}

}  // namespace

CpuInfo detect_cpu_info() {
  CpuInfo info;
  info.model_name = model_name_from_proc_cpuinfo();
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  info.logical_cpus = n > 0 ? static_cast<int>(n)
                             : static_cast<int>(std::thread::hardware_concurrency());
  detect_caches(info);
  detect_hugepages(info);
  detect_isa(info);
  return info;
}

}  // namespace roofline
