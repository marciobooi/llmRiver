#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "compute_bench.hpp"
#include "cpu_info.hpp"
#include "disk_bench.hpp"
#include "hugepage_bench.hpp"
#include "latency_bench.hpp"
#include "ram_bench.hpp"
#include "report.hpp"

namespace {

using roofline::Report;

struct Options {
  std::map<std::string, std::string> kv;

  std::string get(const std::string& key, const std::string& def) const {
    auto it = kv.find(key);
    return it == kv.end() ? def : it->second;
  }
  long get_long(const std::string& key, long def) const {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    try {
      return std::stol(it->second);
    } catch (...) {
      return def;
    }
  }
  bool has(const std::string& key) const { return kv.count(key) > 0; }
};

Options parse_options(int argc, char** argv, int start) {
  Options opts;
  for (int i = start; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.rfind("--", 0) != 0) continue;
    arg = arg.substr(2);
    auto eq = arg.find('=');
    if (eq == std::string::npos) {
      opts.kv[arg] = "1";
    } else {
      opts.kv[arg.substr(0, eq)] = arg.substr(eq + 1);
    }
  }
  return opts;
}

void print_usage(const char* prog) {
  std::cerr
      << "usage: " << prog << " <all|ram|latency|disk|compute|hugepage> [options]\n\n"
      << "This is a machine resistance-matrix profiler -- see docs/PLAN.md.\n"
      << "Rebuild and run it on the actual target machine; numbers taken on\n"
      << "a dev sandbox / VM are not representative.\n\n"
      << "options (all optional, sane defaults):\n"
      << "  --duration-ms=N          time budget per sub-benchmark (default 300)\n"
      << "  --ram-buf-mb=N           per-thread RAM buffer, should exceed L3 (default 512)\n"
      << "  --ram-threads=N          max threads for the RAM sweep (default: nproc)\n"
      << "  --latency-footprint-mb=N pointer-chase footprint, should exceed L3 (default 512)\n"
      << "  --latency-max-streams=N  max independent MLP streams to sweep (default 64)\n"
      << "  --disk-file=PATH         target file for disk bench (default /tmp/roofline_disk_test.bin)\n"
      << "  --disk-size-mb=N         disk test file size (default 512)\n"
      << "  --hugepage-footprint-mb=N hugepage-vs-4K footprint (default 512)\n"
      << "  --json                   print JSON to stdout instead of the human summary\n"
      << "  --out=PATH               also write the JSON report to PATH\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }
  std::string cmd = argv[1];
  Options opts = parse_options(argc, argv, 2);

  int duration_ms = static_cast<int>(opts.get_long("duration-ms", 300));
  int default_threads = static_cast<int>(std::thread::hardware_concurrency());
  if (default_threads <= 0) default_threads = 4;

  bool want_ram = cmd == "all" || cmd == "ram";
  bool want_latency = cmd == "all" || cmd == "latency";
  bool want_disk = cmd == "all" || cmd == "disk";
  bool want_compute = cmd == "all" || cmd == "compute";
  bool want_hugepage = cmd == "all" || cmd == "hugepage";

  if (!want_ram && !want_latency && !want_disk && !want_compute && !want_hugepage) {
    print_usage(argv[0]);
    return 1;
  }

  Report report;
  report.cpu = roofline::detect_cpu_info();

  if (want_ram) {
    size_t buf_bytes = static_cast<size_t>(opts.get_long("ram-buf-mb", 512)) * 1024 * 1024;
    int threads = static_cast<int>(opts.get_long("ram-threads", default_threads));
    std::cerr << "[roofline] ram bench: " << threads << " threads x "
              << (buf_bytes / (1024 * 1024)) << " MB...\n";
    report.ram = roofline::run_ram_bench(buf_bytes, threads, duration_ms);
  }

  if (want_latency) {
    size_t footprint =
        static_cast<size_t>(opts.get_long("latency-footprint-mb", 512)) * 1024 * 1024;
    int max_streams = static_cast<int>(opts.get_long("latency-max-streams", 64));
    std::cerr << "[roofline] latency/MLP bench: " << (footprint / (1024 * 1024))
              << " MB footprint, up to " << max_streams << " streams...\n";
    report.latency = roofline::run_latency_bench(footprint, max_streams, duration_ms);
  }

  if (want_disk) {
    std::string path = opts.get("disk-file", "/tmp/roofline_disk_test.bin");
    uint64_t size_bytes = static_cast<uint64_t>(opts.get_long("disk-size-mb", 512)) * 1024 * 1024;
    std::cerr << "[roofline] disk bench: " << path << " (" << (size_bytes / (1024 * 1024))
              << " MB)...\n";
    report.disk = roofline::run_disk_bench(path, size_bytes, duration_ms);
  }

  if (want_compute) {
    std::cerr << "[roofline] compute bench...\n";
    report.compute = roofline::run_compute_bench(duration_ms);
  }

  if (want_hugepage) {
    size_t footprint =
        static_cast<size_t>(opts.get_long("hugepage-footprint-mb", 512)) * 1024 * 1024;
    std::cerr << "[roofline] hugepage bench: " << (footprint / (1024 * 1024))
              << " MB footprint...\n";
    report.hugepage = roofline::run_hugepage_bench(footprint, duration_ms);
  }

  if (opts.has("out")) {
    std::ofstream f(opts.get("out", ""));
    roofline::write_json(f, report);
  }

  if (opts.has("json")) {
    roofline::write_json(std::cout, report);
  } else {
    roofline::write_human(std::cout, report);
  }

  return 0;
}
