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
  speed)
    # 35.2 tok/s short, 29.8 at 4K, 14.4 at 16K -- fastest at every depth
    # measured (docs/reports/architecture-comparison-2026-08-19.md).
    # ~1B active params of 8B, so only ~1.25 GB is read per token, plus
    # hybrid conv/attention keeps long-context cost down. Trade-off: it is
    # an 8B model, weaker than the 30B MoEs on hard reasoning, and it is a
    # reasoning model that spends tokens thinking before answering.
    MODEL_DIR=~/llmriver/models/lfm25-8b
    MODEL_FILE=LFM2.5-8B-A1B-Q4_K_M.gguf
    MODEL_URL='https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF/resolve/main/LFM2.5-8B-A1B-Q4_K_M.gguf'
    PARALLEL=1
    CTX=32768
    ;;
  single)
    # 20.8 tok/s. Q3_K_M rather than Q2_K: measured wikitext perplexity is
    # 8.29 vs Q4_K_M's 8.04 (+3.1%, error bars overlap, i.e. no detectable
    # quality loss) while Q2_K costs +9.5% with non-overlapping error bars.
    # Q3_K_M buys most of the speed for none of the measurable quality.
    MODEL_DIR=~/llmriver/models/qwen3-30b-q3km
    MODEL_FILE=Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf
    MODEL_URL='https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf'
    PARALLEL=1
    CTX=8192
    ;;
  longctx)
    # For prompts beyond ~13K tokens IF you prefer a 30B-class model there.
    # DeepSeek-V2-Lite (MoE + MLA) measured 12.2 tok/s at 4K and 3.8 at 16K.
    # Note 'speed' beats it at every depth (29.8 / 14.4) with a smaller model.
    MODEL_DIR=~/llmriver/models/deepseek-v2-lite
    MODEL_FILE=DeepSeek-V2-Lite-Chat-Q4_K_M.gguf
    MODEL_URL='https://huggingface.co/second-state/DeepSeek-V2-Lite-Chat-GGUF/resolve/main/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf'
    PARALLEL=1
    CTX=32768
    ;;
  serving)
    # 45.5 tok/s aggregate across 16 concurrent requests. Q4_K_M rather than
    # Q2_K: batching pushes decode compute-bound, where Q2_K's cheaper bytes
    # stop paying and its pricier dequant starts costing (it loses by 7% at
    # this batch size). Better quality too.
    MODEL_DIR=~/llmriver/models/qwen3-30b-a3b-q4km
    MODEL_FILE=Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf
    MODEL_URL='https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf'
    PARALLEL=16
    CTX=65536
    ;;
  *)
    echo "usage: $0 [speed|single|longctx|serving]" >&2
    echo "  speed   LFM2.5-8B-A1B  - fastest at every depth (35/30/14 tok/s)" >&2
    echo "  single  Qwen3-30B Q3_K_M - best quality (20.8 tok/s)" >&2
    echo "  longctx DeepSeek-V2-Lite - 30B-class with MLA (needs download)" >&2
    echo "  serving Qwen3-30B Q4_K_M - 16 concurrent (needs download)" >&2
    exit 1
    ;;
esac

# Only 'speed' and 'single' models are kept on disk; the rest were deleted to
# reclaim ~107GB. Fail with the URL rather than a cryptic docker mount error.
if [ ! -f "${MODEL_DIR/#\~/$HOME}/$MODEL_FILE" ]; then
  echo "model not present: $MODEL_DIR/$MODEL_FILE" >&2
  echo "download it with:" >&2
  echo "  mkdir -p $MODEL_DIR && curl -L --retry 5 -o $MODEL_DIR/$MODEL_FILE \\" >&2
  echo "    '${MODEL_URL:-see docs/reports/architecture-comparison-2026-08-19.md}'" >&2
  exit 1
fi

# No draft model on purpose: speculative decoding makes this MoE *slower*
# (18.5 -> 10.0 tok/s as draft depth grows), because verifying N draft tokens
# reads the union of N tokens' expert sets rather than the same weights.
exec docker run -d --name llmriver-server \
  -p 127.0.0.1:8080:8080 \
  -v "$MODEL_DIR":/moe:ro \
  --entrypoint llama-server llmriver-llamacpp:b10502-ppl \
  -m /moe/"$MODEL_FILE" \
  --host 0.0.0.0 --port 8080 \
  -t 6 -fa 1 -c "$CTX" --parallel "$PARALLEL"
