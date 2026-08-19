# llamacpp

`Dockerfile` builds vanilla, unmodified upstream `llama.cpp` (pinned
tag) — this is deliberately **not** part of this project's own ideas,
it's the "before" baseline that docs/PLAN.md Step 2's claims get
measured against. No llmRiver code belongs in that image.

`patches/` holds the actual llmRiver-authored changes tested against
that baseline, one patch file per attempt, applied on top of the same
pinned tag's source. See docs/PLAN.md Step 2 for the full writeup of
each attempt (hypothesis, result, root cause) — `patches/0001-*` is the
first one, killed, with the reasoning why.

## Build (on target hardware)

Like `tools/roofline`, `GGML_NATIVE=ON` resolves to the actual build-host
CPU, so build this on the machine being benchmarked:

```
docker build -t llmriver-llamacpp:b10499 .
```

## Run a benchmark

```
docker run --rm --network none --name llmriver-llamacpp-bench \
  -v ~/llmriver/models/<model-dir>:/models:ro \
  llmriver-llamacpp:b10499 \
  -m /models/<model-file>.gguf \
  -p 512 -n 128 -t 6 -fa 1 \
  -o json
```

`--network none` + `--rm` + a dedicated model bind mount (read-only):
same isolation pattern as `tools/roofline` — no port, no persistent
container, no shared volumes with anything else on the host.

## Models on `mercury-hetzner`

Not tracked in git (multi-GB binaries) — live only on the server at
`~/llmriver/models/`. Re-download from the official Hugging Face repo
if the host is rebuilt; SHA256 below is what was verified at download
time.

| model | path on server | sha256 |
|---|---|---|
| Qwen2.5-7B-Instruct Q4_K_M | `~/llmriver/models/qwen2.5-7b-instruct-q4_k_m/` | shard1 `dfce12e3...0580db` / shard2 `539cf93f...df3d72a` |
| Qwen2.5-7B-Instruct Q3_K_M | `~/llmriver/models/qwen2.5-7b-instruct-q3_k_m/` | `a96b1617...21cca5e` |
| Qwen2.5-0.5B-Instruct Q8_0 (speculative-decoding draft) | `~/llmriver/models/qwen2.5-0.5b-instruct-q8_0/` | `ca59ca7f...9ff76844e` |
| Qwen2.5-7B-Instruct Q4_0 | `~/llmriver/models/qwen2.5-7b-instruct-q4_0/` | 2 shards |
| Qwen2.5-0.5B-Instruct Q4_K_M (faster draft) | `~/llmriver/models/qwen2.5-0.5b-instruct-q4_k_m/` | — |
| **Qwen3-30B-A3B-Instruct-2507 Q4_K_M (MoE — fastest)** | `~/llmriver/models/qwen3-30b-a3b-q4km/` | `6c997b8a...6c4774d0` (verified) |
| Qwen3-0.6B Q8_0 (MoE draft — **not recommended**, see below) | `~/llmriver/models/qwen3-0.6b-q8/` | — |

Source: `Qwen/Qwen2.5-7B-Instruct-GGUF` / `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
on Hugging Face. Q4_K_M is split into two `gguf-split`-convention shards
(`-00001-of-00002` / `-00002-of-00002`) by the upstream repo; llama.cpp
auto-detects and loads both when given the first shard's path. Q3_K_M
and the draft model are single files.

## Recommended flags on `mercury-hetzner` (AMD Ryzen 5 3600, 6c/12t)

`-t 6 -fa 1` (6 threads = physical core count, flash attention on).
A thread/flash-attention sweep (`docs/reports/llamacpp-tuning-sweep-*.json`)
found this beats the naive `-t 12` default by ~6%:

| threads | flash_attn=0 | flash_attn=1 |
|---|---|---|
| 4  | 9.04 tok/s | 9.15 tok/s |
| 6  | 9.04 tok/s | **9.15 tok/s** |
| 8  | 8.95 tok/s | 9.07 tok/s |
| 12 | 8.64 tok/s | 8.72 tok/s |

(tg64 decode, Qwen2.5-7B-Instruct Q4_K_M.) Using all 12 logical threads
is *slower* than using the 6 physical cores — decode here is bandwidth-
bound (§III.1 of the critical-analysis doc), and SMT siblings share a
physical core's line-fill buffers rather than adding memory-level
parallelism, so the extra threads only add dispatch/sync overhead.
This tuned number is already ~97% of the ~9.4 tok/s ceiling the
roofline's measured RAM bandwidth predicts for this model size — see
docs/PLAN.md Step 2 for the full baseline writeup and what would
actually need to change to beat it.

## Which model to run — depends on your context length

There is no single fastest model here. Measured decode tok/s by KV depth
(`-t 6 -fa 1`, llama.cpp b10502):

| depth | MoE 30B Q2_K | Gemma 3 12B (sliding-window) |
|---|---|---|
| 0 | **26.77** | 4.97 |
| 4,096 | **10.25** | 4.58 |
| 8,192 | **6.25** | 3.81 |
| 16,384 | 2.34 | **3.12** |

- **Short prompts / most chat turns:** MoE 30B Q2_K, by up to 5.4x.
- **Past ~12-13K tokens of context:** Gemma 3 12B. The MoE decays -91%
  across this range against Gemma 3's -37%, because expert sparsity does
  not apply to attention and the MoE carries 48 attention layers.
- **Past ~16K:** everything on this host is 2-3 tok/s. That is below
  comfortable interactive speed no matter which model you pick.

## Running it as an API

`serve.sh` starts the winning config as an OpenAI-compatible HTTP server:

```
./serve.sh single    # 25.7 tok/s, one user at a time (MoE Q2_K)
./serve.sh serving   # 45.5 tok/s aggregate, 16 concurrent (MoE Q4_K_M)
docker rm -f llmriver-server   # stop it
```

Needs the `llmriver-llamacpp:b10499-server` image (the `llama-server`
target isn't in the base Dockerfile's target list; see `serve.sh`).

**It binds to 127.0.0.1 on purpose.** This host has a public IP and
`llama-server` ships no authentication — publishing it on 0.0.0.0 would
put an open LLM endpoint on the internet. Reach it from your laptop with
a tunnel instead:

```
ssh -i ~/.ssh/mercury_admin -L 8080:127.0.0.1:8080 mercury@65.109.96.74
```

## Fastest known configurations on this host

Use these unless you have a reason not to. Full reasoning in
docs/PLAN.md Step 2 and
[docs/reports/moe-sparsity-the-real-unlock-2026-08-19.md](../../docs/reports/moe-sparsity-the-real-unlock-2026-08-19.md).

**Single user (18.7 tok/s, 2.05x the dense-7B baseline):**

```
docker run --rm --network none \
  -v ~/llmriver/models/qwen3-30b-a3b-q4km:/moe:ro \
  --entrypoint llama-cli llmriver-llamacpp:b10499 \
  -m /moe/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
  -st -t 6 -fa 1 -n 128 -p 'your prompt'
```

**Many concurrent users (45.5 tok/s aggregate, 5.0x baseline):** same
model, served with 16 parallel slots.

Do **not** add a draft model (`-md`) to the MoE — speculation makes it
*slower* (18.5 -> 10.0 as draft depth grows). Speculation only helps the
dense models here.

## What actually beat the baseline

Flag-tuning caps out around 3% (see above — vanilla llama.cpp is
already near the physical bandwidth wall). Two things that go further,
each with a real cost, not a free lunch — full detail and repro
commands in docs/PLAN.md Step 2, "Attempt 2" / "Attempt 3":

- **Smaller quantization** (Q3_K_M instead of Q4_K_M): 9.15 -> 11.13
  tok/s (+21.6%). Costs model quality/accuracy, not engineering effort.
- **Speculative decoding** (`-md <draft.gguf> --spec-type draft-simple`,
  Qwen2.5-0.5B-Instruct Q8_0 as draft): 9.0 -> 13.1 tok/s (+45%) on
  predictable/technical prompts, only +10% on creative/unpredictable
  ones. Costs consistency (gain is content-dependent) plus the memory/
  complexity of running a second model.
