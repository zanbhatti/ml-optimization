#!/bin/sh
# Thinking-mode sweep. Usage: ./run_thinking.sh <family> <n> <precisions...>
# Resumable: rerun the same command after a stop/sleep; finished puzzles are skipped.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
family=$1; n=$2; shift 2
for p in "$@"; do
  caffeinate -i .venv/bin/python src/thinking.py "$p" --family "$family" --n "$n"
done
