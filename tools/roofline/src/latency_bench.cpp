#include "latency_bench.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "chase.hpp"
#include "util.hpp"

namespace roofline {
namespace {

using Node = ChaseNode;

std::atomic<uint64_t> g_sink{0};

Node* build_chase_array(size_t footprint_bytes, size_t& out_count) {
  size_t count = footprint_bytes / sizeof(Node);
  if (count < 2) count = 2;
  Node* arr = static_cast<Node*>(std::aligned_alloc(64, count * sizeof(Node)));
  wire_single_cycle(arr, count, /*seed=*/0x5eed5eedULL);
  out_count = count;
  return arr;
}

double measure_dependent_latency(const Node* arr, int duration_ms) {
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  uint32_t cur = 0;
  uint64_t hops = 0;
  uint64_t start = now_ns();
  do {
    // Unroll a bit so the loop-control overhead doesn't dominate a
    // ~1-5ns-per-hop measurement, while each load still strictly depends
    // on the previous one.
    for (int u = 0; u < 64; ++u) cur = arr[cur].next;
    hops += 64;
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;
  g_sink.fetch_add(cur, std::memory_order_relaxed);
  return static_cast<double>(elapsed) / static_cast<double>(hops);
}

// K independent cursors advanced in round-robin within one thread. This is
// the software mechanism for exposing memory-level parallelism to a single
// core: K loads with no dependency on each other can be outstanding at
// once (up to whatever the core's miss-handling hardware allows).
double measure_mlp_bandwidth(const Node* arr, size_t count, int streams,
                              int duration_ms) {
  uint64_t duration_ns = static_cast<uint64_t>(duration_ms) * 1000000ull;
  std::vector<uint32_t> cursors(static_cast<size_t>(streams));
  for (int k = 0; k < streams; ++k) {
    cursors[static_cast<size_t>(k)] =
        static_cast<uint32_t>((count / static_cast<size_t>(streams)) * static_cast<size_t>(k)) % count;
  }
  uint64_t hops = 0;
  uint64_t start = now_ns();
  do {
    for (int rep = 0; rep < 16; ++rep) {
      for (int k = 0; k < streams; ++k) {
        cursors[static_cast<size_t>(k)] = arr[cursors[static_cast<size_t>(k)]].next;
      }
    }
    hops += static_cast<uint64_t>(16 * streams);
  } while (now_ns() - start < duration_ns);
  uint64_t elapsed = now_ns() - start;
  uint64_t chk = 0;
  for (auto c : cursors) chk += c;
  g_sink.fetch_add(chk, std::memory_order_relaxed);

  double bytes = static_cast<double>(hops) * sizeof(Node);
  double seconds = static_cast<double>(elapsed) / 1e9;
  return seconds > 0 ? (bytes / 1e9) / seconds : 0.0;
}

}  // namespace

LatencyBenchResult run_latency_bench(size_t footprint_bytes, int max_streams,
                                      int duration_ms) {
  LatencyBenchResult result;
  size_t count = 0;
  Node* arr = build_chase_array(footprint_bytes, count);
  result.footprint_bytes = count * sizeof(Node);

  result.dependent_chain_latency_ns =
      measure_dependent_latency(arr, duration_ms);

  static const int candidates[] = {1, 2, 4, 8, 16, 32, 64, 128};
  for (int k : candidates) {
    if (k > max_streams) break;
    if (static_cast<size_t>(k) > count) break;
    double gbs = measure_mlp_bandwidth(arr, count, k, duration_ms);
    result.mlp_scaling.push_back({k, gbs});
  }

  for (size_t i = 0; i + 1 < result.mlp_scaling.size(); ++i) {
    double a = result.mlp_scaling[i].bandwidth_gbs;
    double b = result.mlp_scaling[i + 1].bandwidth_gbs;
    if (a <= 0) continue;
    if ((b - a) / a < 0.05) {
      result.mlp_plateau_streams = result.mlp_scaling[i].streams;
      result.mlp_plateau_bandwidth_gbs = a;
      break;
    }
  }
  if (result.mlp_plateau_streams == 0 && !result.mlp_scaling.empty()) {
    result.mlp_plateau_streams = result.mlp_scaling.back().streams;
    result.mlp_plateau_bandwidth_gbs = result.mlp_scaling.back().bandwidth_gbs;
  }

  if (result.dependent_chain_latency_ns > 0) {
    double bytes_in_flight =
        result.mlp_plateau_bandwidth_gbs * result.dependent_chain_latency_ns;
    result.implied_outstanding_requests = bytes_in_flight / sizeof(Node);
  }

  std::free(arr);
  return result;
}

}  // namespace roofline
