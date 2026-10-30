"""Speed and memory per model size and precision: what quantization buys you.

For each (family, precision) the model is loaded in its own subprocess (clean peak-memory
numbers, nothing left resident between models) and timed on two phases:

  prefill  - reading a prompt of L tokens in one pass (all tokens processed together)
  decode   - generating new tokens one at a time

at two context lengths (512 and 4096 tokens: a short prompt, and the length a thinking run
reaches). EOS is masked during timing so every run decodes the full token count. Each
measurement is repeated and the median kept. Results append to a JSONL file, one row per
model, so the sweep resumes after a stop.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from models import FAMILIES, PRECISION_NAMES, model_path
from proofwriter_test import ROOT, build_or_load_eval_set, build_prompt

OUT = ROOT / "results" / "benchmark" / "m3pro-18gb.jsonl"
CONTEXTS = (512, 4096)
DECODE_TOKENS = 128
REPEATS = 3


def weights_dir(family, precision):
    if precision == "bf16":
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(FAMILIES[family], local_files_only=True))
    return Path(model_path(family, precision))


def disk_bytes(directory):
    return sum(f.stat().st_size for f in Path(directory).glob("*.safetensors"))


def prompt_tokens(tokenizer, length):
    """Real task text (ProofWriter prompts back to back), cut to exactly `length` tokens."""
    text = "\n\n".join(build_prompt(e) for e in build_or_load_eval_set()[:200])
    ids = tokenizer.encode(text)
    assert len(ids) >= length, "not enough prompt text"
    return ids[:length]


def measure(family, precision, contexts=CONTEXTS, decode_tokens=DECODE_TOKENS, repeats=REPEATS):
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    mx.set_cache_limit(int(0.5 * 2**30))
    directory = weights_dir(family, precision)
    model, tokenizer = load(str(directory))
    resident = mx.get_active_memory()
    banned = list(tokenizer.eos_token_ids)

    def no_eos(tokens, logits):
        logits[:, banned] = -float("inf")
        return logits

    def once(ids, n):
        mx.reset_peak_memory()
        last = None
        for last in stream_generate(model, tokenizer, ids, max_tokens=n, logits_processors=[no_eos]):
            pass
        mx.clear_cache()
        return last

    once(prompt_tokens(tokenizer, 64), 16)  # warm-up: compile kernels, page weights in
    row = {"family": family, "precision": precision, "disk_gb": disk_bytes(directory) / 1e9,
           "resident_gb": resident / 1e9, "decode_tokens": decode_tokens, "repeats": repeats}
    for length in contexts:
        ids = prompt_tokens(tokenizer, length)
        runs = [once(ids, decode_tokens) for _ in range(repeats)]
        assert all(r.generation_tokens == decode_tokens for r in runs)
        row[f"prefill_tps_{length}"] = statistics.median(r.prompt_tps for r in runs)
        row[f"decode_tps_{length}"] = statistics.median(r.generation_tps for r in runs)
        row[f"peak_gb_{length}"] = max(r.peak_memory for r in runs)
    return row


def done_keys(path):
    if not path.exists():
        return set()
    return {(r["family"], r["precision"]) for r in map(json.loads, path.read_text().splitlines()) if r}


def available(family, precision):
    try:
        return disk_bytes(weights_dir(family, precision)) > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b"],
                        choices=sorted(FAMILIES))
    parser.add_argument("--precisions", nargs="+", default=["bf16", "8bit", "4bit", "3bit", "gptq4"],
                        choices=PRECISION_NAMES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--one", nargs=2, metavar=("FAMILY", "PRECISION"), help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help="One context (512), 32 tokens, 1 repeat")
    args = parser.parse_args()

    if args.one:  # child process: measure one model, print one JSON line
        kwargs = {"contexts": (512,), "decode_tokens": 32, "repeats": 1} if args.smoke else {}
        print("RESULT " + json.dumps(measure(*args.one, **kwargs)), flush=True)
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(args.out)
    for family in args.families:
        for precision in args.precisions:
            if (family, precision) in skip:
                print(f"{family} {precision}: done, skipping", flush=True)
                continue
            if not available(family, precision):
                print(f"{family} {precision}: no local weights, skipping (this script never downloads or builds)",
                      flush=True)
                continue
            t0 = time.time()
            cmd = [sys.executable, __file__, "--one", family, precision] + (["--smoke"] if args.smoke else [])
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            row = json.loads(next(line for line in out.splitlines() if line.startswith("RESULT "))[7:])
            with args.out.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            ctx = min((k for k in row if k.startswith("decode_tps_")), key=lambda k: int(k.split("_")[-1]))
            print(f"{family} {precision}: decode {row[ctx]:.0f} tok/s, prefill "
                  f"{row[ctx.replace('decode', 'prefill')]:.0f} tok/s, peak "
                  f"{max(v for k, v in row.items() if k.startswith('peak_gb')):.2f} GB "
                  f"({time.time() - t0:.0f}s)", flush=True)
            time.sleep(5)  # let the chip cool a little between models


if __name__ == "__main__":
    main()
