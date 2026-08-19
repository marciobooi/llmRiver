#!/bin/bash
# Starts the fastest known llmriver config as an OpenAI-compatible HTTP API.
#
# Bound to 127.0.0.1 deliberately: this host has a public IP and llama-server
# has no authentication. Do NOT change --publish to 0.0.0.0 without putting
# an authenticating reverse proxy in front of it. Reach it remotely with an
# SSH tunnel instead:
#   ssh -i ~/.ssh/mercury_admin -L 8080:127.0.0.1:8080 mercury@65.109.96.74
#
# Requires the image built by tools/llamacpp/Dockerfile plus the llama-server
# target (see README.md). Stop with: docker rm -f llmriver-server
set -euo pipefail

MODE="${1:-single}"

case "$MODE" in
  single)
    # 25.7 tok/s — fastest for one user at a time (docs/PLAN.md Step 2,
    # Attempt 11). Q2_K wins here because batch=1 decode is bandwidth-bound.
    MODEL_DIR=~/llmriver/models/qwen3-30b-q2k
    MODEL_FILE=Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf
    PARALLEL=1
    CTX=8192
    ;;
  serving)
    # 45.5 tok/s aggregate across 16 concurrent requests. Q4_K_M rather than
    # Q2_K: batching pushes decode compute-bound, where Q2_K's cheaper bytes
    # stop paying and its pricier dequant starts costing (it loses by 7% at
    # this batch size). Better quality too.
    MODEL_DIR=~/llmriver/models/qwen3-30b-a3b-q4km
    MODEL_FILE=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
    PARALLEL=16
    CTX=65536
    ;;
  *)
    echo "usage: $0 [single|serving]" >&2
    exit 1
    ;;
esac

# No draft model on purpose: speculative decoding makes this MoE *slower*
# (18.5 -> 10.0 tok/s as draft depth grows), because verifying N draft tokens
# reads the union of N tokens' expert sets rather than the same weights.
exec docker run -d --name llmriver-server \
  -p 127.0.0.1:8080:8080 \
  -v "$MODEL_DIR":/moe:ro \
  --entrypoint llama-server llmriver-llamacpp:b10499-server \
  -m /moe/"$MODEL_FILE" \
  --host 0.0.0.0 --port 8080 \
  -t 6 -fa 1 -c "$CTX" --parallel "$PARALLEL"
