#include "compute_bench.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "util.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace roofline {
namespace {

volatile double g_sink_d = 0.0;
volatile int64_t g_sink_i = 0;

#if defined(__AVX512F__)
double measure_fp32_fma(int duration_ms) {
  constexpr size_t kFloats = 4096;  // 16KB: comfortably resident in L1d
  constexpr size_t kVecs = kFloats / 16;
  static_assert(kVecs % 8 == 0, "grouping assumes a multiple of 8 vectors");

  float* buf = static_cast<float*>(std::aligned_alloc(64, kFloats * sizeof(float)));
  for (size_t i = 0; i < kFloats; ++i) buf[i] = 1.0f + static_cast<float>(i % 7) * 1e-3f;

  __m512 acc[8];
  for (auto& a : acc) a = _mm512_setzero_ps();

  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  uint64_t passes = 0;
  uint64_t start = now_ns();
  do {
    for (size_t i = 0; i < kVecs; i += 8) {
      for (int j = 0; j < 8; ++j) {
        __m512 v = _mm512_load_ps(&buf[(i + static_cast<size_t>(j)) * 16]);
        acc[j] = _mm512_fmadd_ps(v, v, acc[j]);
      }
    }
    ++passes;
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;

  double sum = 0.0;
  for (auto& a : acc) sum += _mm512_reduce_add_ps(a);
  g_sink_d = sum;

  double total_flops = static_cast<double>(passes) * static_cast<double>(kFloats) * 2.0;
  double seconds = static_cast<double>(elapsed) / 1e9;
  std::free(buf);
  return seconds > 0 ? (total_flops / 1e9) / seconds : 0.0;
}
#endif

#if defined(__AVX512VNNI__)
double measure_int8_vnni(int duration_ms) {
  constexpr size_t kBytes = 4096;  // resident in L1d
  constexpr size_t kVecs = kBytes / 64;
  static_assert(kVecs % 8 == 0, "grouping assumes a multiple of 8 vectors");

  uint8_t* a = static_cast<uint8_t*>(std::aligned_alloc(64, kBytes));
  int8_t* b = static_cast<int8_t*>(std::aligned_alloc(64, kBytes));
  for (size_t i = 0; i < kBytes; ++i) {
    a[i] = static_cast<uint8_t>(i % 128);
    b[i] = static_cast<int8_t>((i % 64) - 32);
  }

  __m512i acc[8];
  for (auto& v : acc) v = _mm512_setzero_si512();

  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  uint64_t passes = 0;
  uint64_t start = now_ns();
  do {
    for (size_t i = 0; i < kVecs; i += 8) {
      for (int j = 0; j < 8; ++j) {
        __m512i va = _mm512_load_si512(&a[(i + static_cast<size_t>(j)) * 64]);
        __m512i vb = _mm512_load_si512(&b[(i + static_cast<size_t>(j)) * 64]);
        acc[j] = _mm512_dpbusd_epi32(acc[j], va, vb);
      }
    }
    ++passes;
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;

  int64_t sum = 0;
  for (auto& v : acc) {
    alignas(64) int32_t lanes[16];
    _mm512_store_si512(lanes, v);
    for (int k = 0; k < 16; ++k) sum += lanes[k];
  }
  g_sink_i = sum;

  // Each _mm512_dpbusd_epi32 does 16 lanes x (4 muls + 4 adds) = 128 ops.
  double total_ops =
      static_cast<double>(passes) * static_cast<double>(kVecs) * 128.0;
  double seconds = static_cast<double>(elapsed) / 1e9;
  std::free(a);
  std::free(b);
  return seconds > 0 ? (total_ops / 1e9) / seconds : 0.0;
}
#endif

}  // namespace

ComputeBenchResult run_compute_bench(int duration_ms) {
  ComputeBenchResult result;

#if defined(__AVX512F__)
  if (__builtin_cpu_supports("avx512f")) {
    result.fp32_fma_available = true;
    result.fp32_fma_gflops = measure_fp32_fma(duration_ms);
  }
#endif

#if defined(__AVX512VNNI__)
  if (__builtin_cpu_supports("avx512vnni")) {
    result.int8_vnni_available = true;
    result.int8_vnni_gops = measure_int8_vnni(duration_ms);
  }
#endif

  return result;
}

}  // namespace roofline
