#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace roofline {

// One cache line. `next` is the only field that matters; the padding
// forces every hop to touch a full, distinct cache line instead of
// several logical nodes sharing one line.
struct alignas(64) ChaseNode {
  uint32_t next;
  uint8_t pad[60];
};

// Sattolo's algorithm: produces a permutation of [0, count) that is a
// single N-cycle (no fixed points, no sub-cycles), unlike a plain
// Fisher-Yates shuffle which can produce many short cycles a smart
// prefetcher could latch onto.
inline std::vector<uint32_t> build_single_cycle_permutation(size_t count, uint64_t seed) {
  std::vector<uint32_t> perm(count);
  for (size_t i = 0; i < count; ++i) perm[i] = static_cast<uint32_t>(i);
  std::mt19937_64 rng(seed);
  for (size_t i = count - 1; i >= 1; --i) {
    std::uniform_int_distribution<size_t> dist(0, i - 1);
    size_t j = dist(rng);
    std::swap(perm[i], perm[j]);
  }
  return perm;
}

// Wires `arr[0..count)` into a single-cycle pointer-chase: starting at any
// node and following `.next` visits every node exactly once before
// repeating.
inline void wire_single_cycle(ChaseNode* arr, size_t count, uint64_t seed) {
  std::vector<uint32_t> perm = build_single_cycle_permutation(count, seed);
  for (size_t k = 0; k < count; ++k) {
    uint32_t from = perm[k];
    uint32_t to = perm[(k + 1) % count];
    arr[from].next = to;
  }
}

}  // namespace roofline
