#!/bin/sh
# Night 2 ([ACT] in the brainstorm log, cleared 2026-10-31): Qwen3-1.7B thinking, 180 items, 4,096 tokens.
# GPTQ runs are omitted: building 1.7B GPTQ with mlx_lm exceeded the 12 GB abort limit (13.3 GB at layer 1).
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
S="--temp 0.6 --top-p 0.95 --top-k 20 --seed 1"
caffeinate -i .venv/bin/python src/thinking.py 4bit --family qwen3-1.7b --n 180
caffeinate -i .venv/bin/python src/thinking.py 4bit --family qwen3-1.7b --n 180 $S
caffeinate -i .venv/bin/python src/thinking.py 8bit --family qwen3-1.7b --n 180
