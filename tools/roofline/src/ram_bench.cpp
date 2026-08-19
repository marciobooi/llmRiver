#include "ram_bench.hpp"

#include <emmintrin.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "util.hpp"

namespace roofline {
namespace {

struct AlignedBuffer {
  uint8_t* ptr = nullptr;
  size_t bytes = 0;

  AlignedBuffer() = default;
  explicit AlignedBuffer(size_t n) {
    bytes = (n + 63) & ~static_cast<size_t>(63);
    ptr = static_cast<uint8_t*>(std::aligned_alloc(64, bytes));
    if (ptr) std::memset(ptr, 0x5A, bytes);
  }
  ~AlignedBuffer() { std::free(ptr); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  AlignedBuffer(AlignedBuffer&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
    o.ptr = nullptr;
    o.bytes = 0;
  }
};

// Escape hatch so the read kernel's accumulation can't be dead-code
// eliminated. Contention on this atomic across threads is negligible next
// to the multi-hundred-MB streaming loop it's fed from once per pass.
std::atomic<uint64_t> g_sink{0};

uint64_t read_kernel(const uint8_t* buf, size_t bytes, uint64_t start_ns,
                      uint64_t duration_ns) {
  const uint64_t* p = reinterpret_cast<const uint64_t*>(buf);
  size_t n = bytes / sizeof(uint64_t);
  uint64_t sum = 0;
  uint64_t passes = 0;
  do {
    for (size_t i = 0; i < n; ++i) sum += p[i];
    ++passes;
  } while (now_ns() - start_ns < duration_ns);
  g_sink.fetch_add(sum, std::memory_order_relaxed);
  return passes * bytes;
}

uint64_t write_plain_kernel(uint8_t* buf, size_t bytes, uint64_t start_ns,
                             uint64_t duration_ns, uint64_t pattern) {
  uint64_t* p = reinterpret_cast<uint64_t*>(buf);
  size_t n = bytes / sizeof(uint64_t);
  uint64_t passes = 0;
  do {
    for (size_t i = 0; i < n; ++i) p[i] = pattern + i;
    ++passes;
  } while (now_ns() - start_ns < duration_ns);
  return passes * bytes;
}

uint64_t write_nt_kernel(uint8_t* buf, size_t bytes, uint64_t start_ns,
                          uint64_t duration_ns, uint64_t pattern) {
  __m128i* p = reinterpret_cast<__m128i*>(buf);
  size_t n = bytes / sizeof(__m128i);
  __m128i val = _mm_set1_epi64x(static_cast<long long>(pattern));
  uint64_t passes = 0;
  do {
    for (size_t i = 0; i < n; ++i) _mm_stream_si128(&p[i], val);
    _mm_sfence();
    ++passes;
  } while (now_ns() - start_ns < duration_ns);
  return passes * bytes;
}

enum class Kernel { Read, WritePlain, WriteNt };

std::vector<int> thread_steps(int max_threads) {
  std::vector<int> steps;
  static const int candidates[] = {1,  2,  3,  4,  6,   8,   12, 16,
                                    24, 32, 48, 64, 96, 128};
  for (int c : candidates) {
    if (c <= max_threads) steps.push_back(c);
  }
  if (max_threads > 0 && (steps.empty() || steps.back() != max_threads)) {
    steps.push_back(max_threads);
  }
  return steps;
}

std::vector<ThreadScalingPoint> sweep(Kernel kind,
                                       std::vector<AlignedBuffer>& buffers,
                                       int max_threads, int duration_ms) {
  std::vector<ThreadScalingPoint> out;
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;

  for (int t : thread_steps(max_threads)) {
    std::vector<std::thread> pool;
    std::vector<uint64_t> bytes_done(static_cast<size_t>(t), 0);
    std::atomic<bool> go{false};

    for (int i = 0; i < t; ++i) {
      pool.emplace_back([&, i]() {
        while (!go.load(std::memory_order_acquire)) {
          // spin until every worker is ready to start together
        }
        uint64_t start = now_ns();
        uint8_t* buf = buffers[static_cast<size_t>(i)].ptr;
        size_t bytes = buffers[static_cast<size_t>(i)].bytes;
        uint64_t done = 0;
        switch (kind) {
          case Kernel::Read:
            done = read_kernel(buf, bytes, start, duration_ns);
            break;
          case Kernel::WritePlain:
            done = write_plain_kernel(buf, bytes, start, duration_ns, 0x1234);
            break;
          case Kernel::WriteNt:
            done = write_nt_kernel(buf, bytes, start, duration_ns, 0x1234);
            break;
        }
        bytes_done[static_cast<size_t>(i)] = done;
      });
    }

    uint64_t wall_start = now_ns();
    go.store(true, std::memory_order_release);
    for (auto& th : pool) th.join();
    uint64_t wall_ns = now_ns() - wall_start;

    uint64_t total_bytes = 0;
    for (auto b : bytes_done) total_bytes += b;
    double seconds = static_cast<double>(wall_ns) / 1e9;
    double gbs = seconds > 0 ? (static_cast<double>(total_bytes) / 1e9) / seconds : 0.0;
    out.push_back({t, gbs, t > 0 ? gbs / t : 0.0});
  }
  return out;
}

}  // namespace

RamBenchResult run_ram_bench(size_t buffer_bytes_per_thread, int max_threads,
                              int duration_ms) {
  RamBenchResult result;
  result.buffer_bytes_per_thread = buffer_bytes_per_thread;
  result.max_threads_tested = max_threads;

  std::vector<AlignedBuffer> buffers;
  buffers.reserve(static_cast<size_t>(max_threads));
  for (int i = 0; i < max_threads; ++i) buffers.emplace_back(buffer_bytes_per_thread);

  result.read_scaling = sweep(Kernel::Read, buffers, max_threads, duration_ms);
  result.write_plain_scaling =
      sweep(Kernel::WritePlain, buffers, max_threads, duration_ms);
  result.write_nt_scaling =
      sweep(Kernel::WriteNt, buffers, max_threads, duration_ms);

  if (!result.read_scaling.empty()) {
    result.single_thread_read_gbs = result.read_scaling.front().aggregate_gbs;
  }

  // Saturation point: first step where adding more threads buys <5% more
  // aggregate bandwidth. This is the empirical stand-in for "how many
  // cores does it take to generate enough memory-level parallelism to
  // saturate DRAM" from §III.1 of the analysis.
  for (size_t i = 0; i + 1 < result.read_scaling.size(); ++i) {
    double a = result.read_scaling[i].aggregate_gbs;
    double b = result.read_scaling[i + 1].aggregate_gbs;
    if (a <= 0) continue;
    if ((b - a) / a < 0.05) {
      result.read_saturation_threads = result.read_scaling[i].threads;
      result.read_saturation_gbs = a;
      break;
    }
  }
  if (result.read_saturation_threads == 0 && !result.read_scaling.empty()) {
    result.read_saturation_threads = result.read_scaling.back().threads;
    result.read_saturation_gbs = result.read_scaling.back().aggregate_gbs;
  }

  return result;
}

}  // namespace roofline
