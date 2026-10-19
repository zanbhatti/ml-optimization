#!/bin/sh
# Full 540-item ProofWriter sweep at BF16, 8-bit and 4-bit, then analysis.
# Resumable: rerun this script after a crash/sleep and finished items are skipped.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
for p in bf16 8bit 4bit; do
  caffeinate -i .venv/bin/python src/run_precision.py "$p"
done
.venv/bin/python src/analyze.py
