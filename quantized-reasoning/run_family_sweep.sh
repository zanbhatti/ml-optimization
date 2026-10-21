#!/bin/sh
# Forced-choice sweep across Qwen3 sizes x precisions, then per-family and cross-size analysis.
# Quantized checkpoints are built on first use. Resumable: rerun after a crash/sleep.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
FAMILIES="${FAMILIES:-qwen3-0.6b qwen3-1.7b qwen3-4b}"
for f in $FAMILIES; do
  for p in bf16 8bit 6bit 5bit 4bit 3bit; do
    caffeinate -i .venv/bin/python src/forced_choice.py "$p" --family "$f"
  done
  .venv/bin/python src/analyze_forced_choice.py --family "$f" > /dev/null
done
.venv/bin/python src/size_trend.py $FAMILIES
