#!/bin/zsh
# Validate block-wise GPTQ (gptq4b) against mlx_lm GPTQ (gptq4) on Qwen3-0.6B ([ACT], after night 2).
# Build runs under a memory watchdog: killed if its footprint passes 12 GB. Resumable runners.
set -e
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
PY=.venv/bin/python
F=qwen3-0.6b

if [ ! -d models/Qwen3-0.6B-gptq4b ]; then
  $PY -c "import sys; sys.path.insert(0,'src'); from models import ensure_quantized; ensure_quantized('$F','gptq4b')" &
  parent=$!; peak=0
  while kill -0 $parent 2>/dev/null; do
    pid=$(pgrep -f "gptq_blockwise.py" | head -1)
    if [ -n "$pid" ]; then
      val=$(footprint -p $pid 2>/dev/null | grep -m1 "Footprint:" | sed -E 's/.*Footprint: ([0-9.]+) ([KMG]B).*/\1 \2/')
      num=${val% *}; unit=${val#* }
      case $unit in KB) mb=$(( num / 1024 ));; MB) mb=$num;; GB) mb=$(( num * 1024 ));; *) mb=0;; esac
      mb=${mb%.*}; [ "$mb" -gt "$peak" ] && peak=$mb
      if [ "$mb" -gt 12288 ]; then echo "ABORT: footprint ${mb} MB > 12 GB"; kill $pid $parent; rm -rf models/Qwen3-0.6B-gptq4b; exit 1; fi
    fi
    sleep 2
  done
  wait $parent
  echo "gptq4b build done; peak process footprint ${peak} MB"
fi

$PY src/validate_gptq.py --family $F
caffeinate -i $PY src/forced_choice.py gptq4b --family $F
$PY src/analyze_forced_choice.py --family $F > /dev/null
caffeinate -i $PY src/thinking.py gptq4b --family $F --n 180
$PY src/analyze_thinking.py --family $F > /dev/null
echo "validation complete"
