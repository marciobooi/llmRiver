cmake_minimum_required(VERSION 3.16)
project(roofline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(roofline
  src/main.cpp
  src/util.cpp
  src/cpu_info.cpp
  src/ram_bench.cpp
  src/latency_bench.cpp
  src/disk_bench.cpp
  src/compute_bench.cpp
  src/hugepage_bench.cpp
  src/report.cpp
)

target_include_directories(roofline PRIVATE src)

# -march=native gives us whatever the build machine actually has (this is a
# per-machine profiler, it is meant to be rebuilt on the target box, not
# distributed as a portable binary).
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
  target_compile_options(roofline PRIVATE -march=native)
endif()

find_package(Threads REQUIRED)
target_link_libraries(roofline PRIVATE Threads::Threads)
