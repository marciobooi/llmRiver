#pragma once

#include <cstdint>
#include <string>

namespace roofline {

struct CpuInfo {
  std::string model_name;
  int logical_cpus = 0;
  uint64_t l1d_bytes_per_core = 0;
  uint64_t l2_bytes_per_core = 0;
  uint64_t l3_bytes_total = 0;
  bool has_avx2 = false;
  bool has_avx512f = false;
  bool has_avx512bw = false;
  bool has_avx512vnni = false;
  bool has_avx512_bf16 = false;
  bool thp_available = false;   // transparent huge pages, madvise-able
  bool hugetlb_available = false;  // explicit reserved hugetlb pages free
};

CpuInfo detect_cpu_info();

}  // namespace roofline
