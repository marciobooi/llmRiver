#include "disk_bench.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "util.hpp"

namespace roofline {
namespace {

constexpr size_t kChunkBytes = 4 * 1024 * 1024;
constexpr size_t kBlockAlign = 4096;

struct AlignedBuf {
  uint8_t* ptr = nullptr;
  explicit AlignedBuf(size_t n) {
    ptr = static_cast<uint8_t*>(std::aligned_alloc(kBlockAlign, n));
  }
  ~AlignedBuf() { std::free(ptr); }
  AlignedBuf(const AlignedBuf&) = delete;
};

uint64_t round_up(uint64_t v, uint64_t mult) { return ((v + mult - 1) / mult) * mult; }

bool file_has_size(const std::string& path, uint64_t want) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  return static_cast<uint64_t>(st.st_size) == want;
}

void create_file(const std::string& path, uint64_t size_bytes) {
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return;
  AlignedBuf buf(kChunkBytes);
  if (buf.ptr) {
    std::memset(buf.ptr, 0x37, kChunkBytes);
    uint64_t written = 0;
    while (written < size_bytes) {
      ssize_t n = write(fd, buf.ptr, kChunkBytes);
      if (n <= 0) break;
      written += static_cast<uint64_t>(n);
    }
  }
  close(fd);
}

// Returns -1 if O_DIRECT could not be used.
int try_open_direct(const std::string& path) {
#ifdef O_DIRECT
  int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) return -1;
  AlignedBuf probe(kBlockAlign);
  ssize_t n = pread(fd, probe.ptr, kBlockAlign, 0);
  if (n < 0) {
    close(fd);
    return -1;
  }
  return fd;
#else
  (void)path;
  return -1;
#endif
}

double measure_sequential(int fd, uint64_t file_size, bool o_direct, int duration_ms) {
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  AlignedBuf buf(kChunkBytes);
  if (!buf.ptr) return 0.0;
  uint64_t offset = 0;
  uint64_t total = 0;
  uint64_t start = now_ns();
  do {
    if (offset + kChunkBytes > file_size) offset = 0;
    ssize_t n = pread(fd, buf.ptr, kChunkBytes, static_cast<off_t>(offset));
    if (n <= 0) break;
    total += static_cast<uint64_t>(n);
    offset += static_cast<uint64_t>(n);
    if (!o_direct) posix_fadvise(fd, static_cast<off_t>(offset - static_cast<uint64_t>(n)),
                                  n, POSIX_FADV_DONTNEED);
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;
  double seconds = static_cast<double>(elapsed) / 1e9;
  return seconds > 0 ? (static_cast<double>(total) / 1e9) / seconds : 0.0;
}

struct Random4kResult {
  double iops = 0.0;
  double bandwidth_mbs = 0.0;
  double avg_latency_us = 0.0;
};

Random4kResult measure_random4k(int fd, uint64_t file_size, bool o_direct, int duration_ms) {
  Random4kResult r;
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  AlignedBuf buf(kBlockAlign);
  if (!buf.ptr) return r;

  uint64_t max_block = file_size / kBlockAlign;
  if (max_block == 0) return r;
  std::mt19937_64 rng(0xd15cULL);
  std::uniform_int_distribution<uint64_t> dist(0, max_block - 1);

  uint64_t ops = 0;
  uint64_t start = now_ns();
  do {
    uint64_t offset = dist(rng) * kBlockAlign;
    ssize_t n = pread(fd, buf.ptr, kBlockAlign, static_cast<off_t>(offset));
    if (n <= 0) break;
    ++ops;
    if (!o_direct && (ops % 256 == 0)) {
      posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    }
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;
  double seconds = static_cast<double>(elapsed) / 1e9;
  if (seconds > 0 && ops > 0) {
    r.iops = static_cast<double>(ops) / seconds;
    r.bandwidth_mbs = (static_cast<double>(ops) * kBlockAlign / (1024.0 * 1024.0)) / seconds;
    r.avg_latency_us = (seconds * 1e6) / static_cast<double>(ops);
  }
  return r;
}

}  // namespace

DiskBenchResult run_disk_bench(const std::string& path, uint64_t file_size_bytes,
                                int duration_ms) {
  DiskBenchResult result;
  result.file_path = path;
  uint64_t size = round_up(file_size_bytes, kChunkBytes);
  result.file_size_bytes = size;

  if (!file_has_size(path, size)) {
    create_file(path, size);
  }

  int fd = try_open_direct(path);
  if (fd >= 0) {
    result.used_o_direct = true;
  } else {
    fd = open(path.c_str(), O_RDONLY);
    result.used_o_direct = false;
    result.note =
        "O_DIRECT unavailable on this filesystem; fell back to buffered "
        "reads with posix_fadvise(DONTNEED). Numbers include some page "
        "cache effect -- re-run on a real block device (not overlayfs/tmpfs) "
        "for trustworthy results.";
  }
  if (fd < 0) {
    result.note = "could not open " + path + ": " + std::strerror(errno);
    return result;
  }

  result.sequential_read_gbs =
      measure_sequential(fd, size, result.used_o_direct, duration_ms);

  Random4kResult rnd = measure_random4k(fd, size, result.used_o_direct, duration_ms);
  result.random4k_iops = rnd.iops;
  result.random4k_bandwidth_mbs = rnd.bandwidth_mbs;
  result.random4k_avg_latency_us = rnd.avg_latency_us;

  close(fd);
  return result;
}

}  // namespace roofline
