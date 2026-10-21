"""Resumable ProofWriter run for one precision; one JSONL record per example.

Records are appended and fsynced as each example finishes, so a crash or sleep
loses at most the in-flight example. Rerunning the same command resumes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from proofwriter_test import (DEPTHS, EVAL_PATH, LABELS, MAX_NEW_TOKENS, ROOT,
                              build_or_load_eval_set, build_prompt, normalize_prediction, summarize)

from models import DEFAULT_FAMILY, PRECISION_NAMES, model_path

PRECISIONS = {p: model_path(DEFAULT_FAMILY, p) for p in PRECISION_NAMES}


def interleaved_order(examples):
    """Round-robin over (depth, label) strata so any prefix of the run stays balanced."""
    strata = {(d, label): [] for d in DEPTHS for label in LABELS}
    for index, example in enumerate(examples):
        strata[example["QDep"], example["answer"]].append(index)
    order = []
    for position in range(max(len(v) for v in strata.values())):
        order.extend(v[position] for v in strata.values() if position < len(v))
    return order


def item_key(example):
    return hashlib.sha256(json.dumps([example["id"], example["theory"], example["question"]]).encode()).hexdigest()[:16]


def run_header(model, examples):
    return {"type": "header", "model": model,
            "eval_set_sha256": hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256(json.dumps([build_prompt(e) for e in examples]).encode()).hexdigest(),
            "decoding": {"method": "greedy", "max_new_tokens": MAX_NEW_TOKENS}}


def load_done(path, header):
    """Return {index: record} from an existing run file, refusing mismatched provenance."""
    if not path.exists():
        return None
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not lines or lines[0].get("type") != "header":
        raise ValueError(f"{path} has no header; refusing to resume")
    if {k: lines[0][k] for k in header} != header:
        raise ValueError(f"{path} was produced under different model/eval/prompt settings")
    return {row["index"]: row for row in lines[1:]}


def run(examples, predict, path, header, limit=None, log_every=10):
    done = load_done(path, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = interleaved_order(examples)[:limit]
    with path.open("a") as handle:
        def write(row):
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if done is None:
            write(header)
            done = {}
        todo = [i for i in order if i not in done]
        print(f"{len(order) - len(todo)} done, {len(todo)} to go -> {path}", flush=True)
        start = time.time()
        for n, index in enumerate(todo, 1):
            example = examples[index]
            t0 = time.time()
            raw = predict(build_prompt(example))
            prediction = normalize_prediction(raw)
            row = {"index": index, "key": item_key(example), "id": example["id"],
                   "depth": example["QDep"], "gold": example["answer"],
                   "raw_prediction": raw, "prediction": prediction,
                   "correct": prediction == example["answer"], "seconds": round(time.time() - t0, 3)}
            write(row)
            done[index] = row
            if n % log_every == 0 or n == len(todo):
                rate = (time.time() - start) / n
                print(f"  {n}/{len(todo)}  {rate:.2f}s/example  eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)
    return [done[i] for i in order]


def write_report(records, header, path):
    by_depth = {d: summarize([r for r in records if r["depth"] == d]) for d in DEPTHS}
    report = dict({k: v for k, v in header.items() if k != "type"},
                  by_depth=by_depth, overall=summarize(records), predictions=records)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=sorted(PRECISIONS))
    parser.add_argument("--model", help="Override the model path/repo for this precision tag")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N examples in interleaved order")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cache-limit-gb", type=float, default=0.5, help="MLX buffer-cache cap")
    args = parser.parse_args()

    examples = build_or_load_eval_set()
    model_name = args.model or PRECISIONS[args.precision]
    header = run_header(model_name, examples)
    sub = "smoke" if args.limit else "raw"
    path = args.out_dir / sub / f"proofwriter_{args.precision}.jsonl"

    from inference import run_inference
    import mlx.core as mx
    from mlx_lm import load
    # MLX's buffer cache is unbounded by default and grew to swap-inducing size on an
    # 18 GB machine over 540 variable-length prompts; cap it and free it per example.
    mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
    model, tokenizer = load(model_name)

    def predict(prompt):
        output = run_inference(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
        mx.clear_cache()
        return output
    records = run(examples, predict, path, header, args.limit)
    print(f"peak MLX memory {mx.get_peak_memory() / 2**30:.1f} GB", flush=True)
    overall = summarize(records)
    print(f"{args.precision}: {overall['correct']}/{overall['total']} = {overall['accuracy']:.1%}; "
          f"invalid {overall['invalid']}", flush=True)
    if not args.limit and len(records) == len(examples):
        write_report(records, header, args.out_dir / f"proofwriter_{args.precision}.json")


if __name__ == "__main__":
    main()
