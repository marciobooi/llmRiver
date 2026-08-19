# Vanilla upstream llama.cpp, unmodified — this is the "before" baseline
# this project's own ideas (docs/PLAN.md Step 2) get measured against.
# No llmRiver code goes in this image. Built with GGML_NATIVE=ON, so like
# tools/roofline it must be built ON the machine being benchmarked.

FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin a specific release so "the baseline" means the same binary every
# time this image is rebuilt, not whatever master happens to be that day.
ARG LLAMA_CPP_REF=b10499
WORKDIR /src
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
      https://github.com/ggml-org/llama.cpp.git .

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
    && cmake --build build -j --target llama-cli llama-bench

ENV PATH="/src/build/bin:${PATH}"
WORKDIR /models
ENTRYPOINT ["llama-bench"]
