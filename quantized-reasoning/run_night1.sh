#!/bin/sh
# Night 1 ([ACT] in the brainstorm log, cleared 2026-10-30): Qwen3-1.7B BF16 thinking mode, 180 items.
# (1) greedy, 4,096 tokens; (2) sampled with the 0.6B decoding-check settings, seed 1. Resumable.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
caffeinate -i .venv/bin/python src/thinking.py bf16 --family qwen3-1.7b --n 180
caffeinate -i .venv/bin/python src/thinking.py bf16 --family qwen3-1.7b --n 180 --temp 0.6 --top-p 0.95 --top-k 20 --seed 1
