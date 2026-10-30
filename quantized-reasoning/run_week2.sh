#!/bin/sh
# Week 2: speed/memory benchmark, then the greedy-decoding check on Qwen3-0.6B.
# Everything runs one job at a time (no overlapping models, clean timings) and resumes:
# rerun the same command after a stop/sleep and finished work is skipped.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
RAW=results/thinking/raw/qwen3-0.6b

# 1. Speed/memory: 0.6B/1.7B/4B x BF16/8/4/3-bit (+ GPTQ where built). ~10-15 min.
caffeinate -i $PY src/benchmark.py
$PY src/analyze_benchmark.py > /dev/null

# 2. Does 4-bit still fail to finish when the model samples, as Qwen recommends for
#    thinking (temp 0.6, top-p 0.95, top-k 20), instead of greedy? 2 seeds x BF16/4-bit.
for seed in 1 2; do
  for p in bf16 4bit; do
    caffeinate -i $PY src/thinking.py "$p" --family qwen3-0.6b --n 180 --temp 0.6 --seed "$seed"
  done
done

# 3. Does it finish if given twice the room? Greedy, 8,192 tokens, only the items each
#    precision truncated at 4,096 (greedy is deterministic, so the other items are unchanged).
for p in 4bit gptq4 bf16; do
  caffeinate -i $PY src/thinking.py "$p" --family qwen3-0.6b --n 180 --max-tokens 8192 \
    --only-truncated-from "$RAW/proofwriter_$p.jsonl"
done
