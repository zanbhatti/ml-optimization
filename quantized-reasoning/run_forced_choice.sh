#!/bin/sh
# Forced-choice sweep over BF16 and 8/6/5/4/3-bit (540 items each), then analysis.
# Resumable: rerun after a crash/sleep and finished items are skipped.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
for p in bf16 8bit 6bit 5bit 4bit 3bit; do
  caffeinate -i .venv/bin/python src/forced_choice.py "$p"
done
.venv/bin/python src/analyze_forced_choice.py
