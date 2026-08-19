# Speculative decoding — Qwen2.5-7B-Instruct Q4_K_M target, Qwen2.5-0.5B-Instruct Q8_0 draft

2026-08-19, `mercury-hetzner` (AMD Ryzen 5 3600, 6c/12t), isolated Docker
(`llmriver-llamacpp:b10499`, `--network none`), `llama-cli -st -t 6 -fa 1
-n 128 --ignore-eos`, `--spec-type draft-simple` for the speculative runs.
Mechanism: this project's §IV.5/§II.5 idea ("speculate wide because width
is free") via llama.cpp's built-in `-md`/`--spec-draft-model` support —
no llmRiver code, this is stock upstream functionality.

## Prompt: "Explain the theory of relativity in detail." (predictable/technical)

| condition | run 1 | run 2 | run 3 |
|---|---|---|---|
| baseline (no draft) | 9.0 tok/s | 9.1 tok/s | 9.1 tok/s |
| speculative | 13.1 tok/s | 13.0 tok/s | 13.2 tok/s |

Speedup: ~45% (9.0 -> 13.1 avg).

Draft acceptance rate (from `-v` verbose log, one run, draft depth 3):
78/144 draft tokens accepted across 48 verification steps (54.2%),
producing an average of 2.6 output tokens per forward pass instead of 1.
Realized speedup (45%) is below the naive 2.6x this might suggest,
because verifying a wider batch costs more compute than a single-token
decode, and the draft model's own generation time isn't free — bandwidth
is amortized, not eliminated.

## Prompt: "Write a short story about a robot learning to paint." (creative/unpredictable)

| condition | result |
|---|---|
| baseline (no draft) | 9.0 tok/s |
| speculative | 9.9 tok/s |

Speedup: ~10%.

## Conclusion

Speculative decoding is a real, reproducible win, but the magnitude is
strongly content-dependent: it clears the >20% bar by a wide margin on
predictable/technical text (draft model guesses right often) and falls
well short of it on creative/unpredictable text (draft model guesses
wrong more, more verification passes wasted). Not a fixed multiplier —
report a range, not a single number, for any claim built on this.
