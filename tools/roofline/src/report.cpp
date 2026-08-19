#include "report.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>

#include "util.hpp"

namespace roofline {
namespace {

void write_cpu(JsonWriter& j, const CpuInfo& c) {
  j.begin_object();
  j.kv("model_name", c.model_name);
  j.kv("logical_cpus", c.logical_cpus);
  j.kv("l1d_bytes_per_core", static_cast<uint64_t>(c.l1d_bytes_per_core));
  j.kv("l2_bytes_per_core", static_cast<uint64_t>(c.l2_bytes_per_core));
  j.kv("l3_bytes_total", static_cast<uint64_t>(c.l3_bytes_total));
  j.kv("has_avx2", c.has_avx2);
  j.kv("has_avx512f", c.has_avx512f);
  j.kv("has_avx512bw", c.has_avx512bw);
  j.kv("has_avx512vnni", c.has_avx512vnni);
  j.kv("has_avx512_bf16", c.has_avx512_bf16);
  j.kv("thp_available", c.thp_available);
  j.kv("hugetlb_available", c.hugetlb_available);
  j.end_object();
}

void write_ram(JsonWriter& j, const RamBenchResult& r) {
  j.begin_object();
  j.kv("buffer_bytes_per_thread", static_cast<uint64_t>(r.buffer_bytes_per_thread));
  j.kv("max_threads_tested", r.max_threads_tested);
  j.kv("single_thread_read_gbs", r.single_thread_read_gbs);
  j.kv("read_saturation_threads", r.read_saturation_threads);
  j.kv("read_saturation_gbs", r.read_saturation_gbs);

  auto write_scaling = [&](const char* key, const std::vector<ThreadScalingPoint>& pts) {
    j.key(key);
    j.begin_array();
    for (const auto& p : pts) {
      j.begin_object();
      j.kv("threads", p.threads);
      j.kv("aggregate_gbs", p.aggregate_gbs);
      j.kv("per_thread_gbs", p.per_thread_gbs);
      j.end_object();
    }
    j.end_array();
  };
  write_scaling("read_scaling", r.read_scaling);
  write_scaling("write_plain_scaling", r.write_plain_scaling);
  write_scaling("write_nt_scaling", r.write_nt_scaling);
  j.end_object();
}

void write_latency(JsonWriter& j, const LatencyBenchResult& r) {
  j.begin_object();
  j.kv("footprint_bytes", static_cast<uint64_t>(r.footprint_bytes));
  j.kv("dependent_chain_latency_ns", r.dependent_chain_latency_ns);
  j.kv("mlp_plateau_streams", r.mlp_plateau_streams);
  j.kv("mlp_plateau_bandwidth_gbs", r.mlp_plateau_bandwidth_gbs);
  j.kv("implied_outstanding_requests", r.implied_outstanding_requests);
  j.key("mlp_scaling");
  j.begin_array();
  for (const auto& p : r.mlp_scaling) {
    j.begin_object();
    j.kv("streams", p.streams);
    j.kv("bandwidth_gbs", p.bandwidth_gbs);
    j.end_object();
  }
  j.end_array();
  j.end_object();
}

void write_disk(JsonWriter& j, const DiskBenchResult& r) {
  j.begin_object();
  j.kv("file_path", r.file_path);
  j.kv("file_size_bytes", r.file_size_bytes);
  j.kv("used_o_direct", r.used_o_direct);
  if (!r.note.empty()) j.kv("note", r.note);
  j.kv("sequential_read_gbs", r.sequential_read_gbs);
  j.kv("random4k_iops", r.random4k_iops);
  j.kv("random4k_bandwidth_mbs", r.random4k_bandwidth_mbs);
  j.kv("random4k_avg_latency_us", r.random4k_avg_latency_us);
  j.end_object();
}

void write_compute(JsonWriter& j, const ComputeBenchResult& r) {
  j.begin_object();
  j.kv("fp32_fma_available", r.fp32_fma_available);
  j.kv("fp32_fma_gflops", r.fp32_fma_gflops);
  j.kv("int8_vnni_available", r.int8_vnni_available);
  j.kv("int8_vnni_gops", r.int8_vnni_gops);
  j.end_object();
}

void write_hugepage(JsonWriter& j, const HugepageBenchResult& r) {
  j.begin_object();
  j.kv("ran", r.ran);
  if (!r.note.empty()) j.kv("note", r.note);
  j.kv("footprint_bytes", static_cast<uint64_t>(r.footprint_bytes));
  j.kv("baseline_4k_latency_ns", r.baseline_4k_latency_ns);
  j.kv("hugepage_latency_ns", r.hugepage_latency_ns);
  j.kv("hugepage_confirmed", r.hugepage_confirmed);
  j.kv("latency_reduction_pct", r.latency_reduction_pct);
  j.kv("baseline_4k_seq_gbs", r.baseline_4k_seq_gbs);
  j.kv("hugepage_seq_gbs", r.hugepage_seq_gbs);
  j.kv("seq_gain_pct", r.seq_gain_pct);
  j.end_object();
}

}  // namespace

void write_json(std::ostream& out, const Report& report) {
  JsonWriter j(out);
  j.begin_object();

  std::time_t t = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
  j.kv("generated_at", std::string(buf));

  j.key("cpu");
  write_cpu(j, report.cpu);

  if (report.ram) {
    j.key("ram");
    write_ram(j, *report.ram);
  }
  if (report.latency) {
    j.key("latency");
    write_latency(j, *report.latency);
  }
  if (report.disk) {
    j.key("disk");
    write_disk(j, *report.disk);
  }
  if (report.compute) {
    j.key("compute");
    write_compute(j, *report.compute);
  }
  if (report.hugepage) {
    j.key("hugepage");
    write_hugepage(j, *report.hugepage);
  }

  j.end_object();
  out << "\n";
}

namespace {
std::string gbs(double v) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(2) << v << " GB/s";
  return ss.str();
}
}  // namespace

void write_human(std::ostream& out, const Report& report) {
  out << std::fixed << std::setprecision(2);
  out << "=== machine ===\n";
  out << "  cpu: " << report.cpu.model_name << " (" << report.cpu.logical_cpus
      << " logical cpus)\n";
  out << "  L1d/core: " << report.cpu.l1d_bytes_per_core / 1024 << " KB   "
      << "L2/core: " << report.cpu.l2_bytes_per_core / 1024 << " KB   "
      << "L3 total: " << report.cpu.l3_bytes_total / (1024 * 1024) << " MB\n";
  out << "  isa: avx2=" << report.cpu.has_avx2 << " avx512f=" << report.cpu.has_avx512f
      << " avx512bw=" << report.cpu.has_avx512bw
      << " avx512vnni=" << report.cpu.has_avx512vnni
      << " avx512bf16=" << report.cpu.has_avx512_bf16 << "\n";
  out << "  thp_available=" << report.cpu.thp_available
      << " hugetlb_free_available=" << report.cpu.hugetlb_available << "\n";

  if (report.ram) {
    const auto& r = *report.ram;
    out << "\n=== ram bandwidth (resistance: DRAM) ===\n";
    out << "  single-thread read ceiling: " << gbs(r.single_thread_read_gbs)
        << "  (§III.1: expect this near the line-fill-buffer bound, not full "
           "DRAM bandwidth)\n";
    out << "  read scaling:\n";
    for (const auto& p : r.read_scaling) {
      out << "    " << p.threads << " thread(s): " << gbs(p.aggregate_gbs)
          << " aggregate (" << gbs(p.per_thread_gbs) << "/thread)\n";
    }
    out << "  saturation at " << r.read_saturation_threads
        << " thread(s): " << gbs(r.read_saturation_gbs) << "\n";
    if (!r.write_plain_scaling.empty() && !r.write_nt_scaling.empty()) {
      double plain = r.write_plain_scaling.back().aggregate_gbs;
      double nt = r.write_nt_scaling.back().aggregate_gbs;
      out << "  write (plain, RFO) at max threads: " << gbs(plain) << "\n";
      out << "  write (non-temporal, no RFO) at max threads: " << gbs(nt) << "\n";
      if (plain > 0) {
        out << "  non-temporal gain: " << ((nt - plain) / plain * 100.0) << "%"
            << "  (§III.3.2: RFO theory predicts up to ~2x)\n";
      }
    }
  }

  if (report.latency) {
    const auto& r = *report.latency;
    out << "\n=== random-access latency & implied concurrency (§III.1) ===\n";
    out << "  dependent-chain latency: " << r.dependent_chain_latency_ns << " ns/hop\n";
    out << "  MLP scaling (bandwidth vs. independent in-flight streams):\n";
    for (const auto& p : r.mlp_scaling) {
      out << "    " << p.streams << " stream(s): " << gbs(p.bandwidth_gbs) << "\n";
    }
    out << "  plateau at " << r.mlp_plateau_streams
        << " streams: " << gbs(r.mlp_plateau_bandwidth_gbs) << "\n";
    out << "  implied outstanding requests at plateau: "
        << r.implied_outstanding_requests
        << "  (compare to ~10-16 line-fill-buffers/core cited in the "
           "analysis -- this is a Little's-Law proxy, not a PMU count)\n";
  }

  if (report.disk) {
    const auto& r = *report.disk;
    out << "\n=== disk (resistance: storage) ===\n";
    out << "  file: " << r.file_path << "  (" << r.file_size_bytes / (1024 * 1024)
        << " MB, o_direct=" << r.used_o_direct << ")\n";
    if (!r.note.empty()) out << "  note: " << r.note << "\n";
    out << "  sequential read: " << gbs(r.sequential_read_gbs) << "\n";
    out << "  random 4K read: " << r.random4k_iops << " IOPS, "
        << r.random4k_bandwidth_mbs << " MB/s, " << r.random4k_avg_latency_us
        << " us avg latency\n";
    out << "  (§I.2: a 40GB model over this sequential link would decode at "
        << (r.sequential_read_gbs / 40.0)
        << " tok/s if every token required reading the whole model)\n";
  }

  if (report.compute) {
    const auto& r = *report.compute;
    out << "\n=== compute (resistance: none -- this is the abundant "
           "resource per §V.1) ===\n";
    if (r.fp32_fma_available) {
      out << "  AVX-512 FP32 FMA: " << r.fp32_fma_gflops << " GFLOP/s (L1-resident)\n";
    } else {
      out << "  AVX-512 FP32 FMA: not available on this build/CPU\n";
    }
    if (r.int8_vnni_available) {
      out << "  AVX-512 VNNI INT8: " << r.int8_vnni_gops << " GOP/s (L1-resident)\n";
    } else {
      out << "  AVX-512 VNNI INT8: not available on this build/CPU\n";
    }
  }

  if (report.hugepage) {
    const auto& r = *report.hugepage;
    out << "\n=== TLB / hugepage effect (§III.3.1) ===\n";
    if (!r.ran) {
      out << "  skipped: " << r.note << "\n";
    } else {
      out << "  4K-page latency: " << r.baseline_4k_latency_ns << " ns/hop\n";
      out << "  hugepage-requested latency: " << r.hugepage_latency_ns
          << " ns/hop (confirmed backed by 2MB pages: " << r.hugepage_confirmed << ")\n";
      out << "  latency reduction (random access): " << r.latency_reduction_pct << "%\n";
      out << "  4K-page sequential: " << r.baseline_4k_seq_gbs << " GB/s\n";
      out << "  hugepage sequential: " << r.hugepage_seq_gbs << " GB/s\n";
      out << "  gain (sequential streaming): " << r.seq_gain_pct << "%\n";
      out << "  ^ LLM weight streaming is sequential; use that number, not the\n"
             "    random-access one, to predict decode speedup.\n";
      if (!r.note.empty()) out << "  note: " << r.note << "\n";
    }
  }
}

}  // namespace roofline
