# Build context is this directory (tools/roofline/). -march=native is
# resolved at `docker build` time, so the image must be built ON the
# machine being profiled (per README.md) — never pull a pre-built image
# onto a different box and trust its numbers.

FROM ubuntu:24.04 AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j

FROM ubuntu:24.04 AS runtime
# libstdc++6 is already part of ubuntu:24.04's base; no extra runtime deps.
COPY --from=builder /src/build/roofline /usr/local/bin/roofline
ENTRYPOINT ["/usr/local/bin/roofline"]
CMD ["all"]
