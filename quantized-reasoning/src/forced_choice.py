"""Forced-choice ProofWriter scoring: one forward pass, log-probs of the three label tokens.

Removes answer-format failures from the measurement and yields a continuous
per-item margin, log p(gold) − max log p(wrong label), for higher statistical power.
Resumable exactly like run_precision.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from proofwriter_test import EVAL_PATH, LABELS, ROOT, build_or_load_eval_set, build_prompt
from run_precision import PRECISIONS, interleaved_order, item_key, load_done

SCORING = "forced_choice_v1: first assistant token, full-vocab log-softmax"


def label_token_ids(tokenizer):
    ids = {}
    for label in LABELS:
        tokens = tokenizer.encode(label, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"{label!r} is not a single token: {tokens}")
        ids[label] = tokens[0]
    return ids


def make_scorer(model, tokenizer):
    import mlx.core as mx
    label_ids = label_token_ids(tokenizer)

    def score(prompt):
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                  tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(formatted)
        logits = model(mx.array([ids]))[0, -1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        top = int(mx.argmax(logprobs).item())
        result = {label: float(logprobs[i].item()) for label, i in label_ids.items()}
        mx.clear_cache()
        return result, top in label_ids.values(), len(ids)
    return score, label_ids


def header_for(model_name, examples, label_ids):
    return {"type": "header", "model": model_name, "scoring": SCORING, "label_token_ids": label_ids,
            "eval_set_sha256": hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256(json.dumps([build_prompt(e) for e in examples]).encode()).hexdigest()}


def record_for(index, example, logprobs, top_is_label, n_tokens, seconds):
    gold = example["answer"]
    prediction = max(LABELS, key=lambda label: logprobs[label])
    margin = logprobs[gold] - max(logprobs[label] for label in LABELS if label != gold)
    return {"index": index, "key": item_key(example), "id": example["id"], "depth": example["QDep"],
            "gold": gold, "logprobs": logprobs, "prediction": prediction, "correct": prediction == gold,
            "margin": margin, "top_token_is_label": top_is_label, "prompt_tokens": n_tokens,
            "seconds": round(seconds, 3)}


def run(examples, score, path, header, limit=None, log_every=50):
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
            t0 = time.time()
            logprobs, top_is_label, n_tokens = score(build_prompt(examples[index]))
            row = record_for(index, examples[index], logprobs, top_is_label, n_tokens, time.time() - t0)
            write(row)
            done[index] = row
            if n % log_every == 0 or n == len(todo):
                rate = (time.time() - start) / n
                print(f"  {n}/{len(todo)}  {rate:.2f}s/example  eta {rate * (len(todo) - n) / 60:.1f} min", flush=True)
    return [done[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=sorted(PRECISIONS))
    parser.add_argument("--model", help="Override the model path/repo for this precision tag")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "forced_choice")
    parser.add_argument("--cache-limit-gb", type=float, default=0.5)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
    examples = build_or_load_eval_set()
    model_name = args.model or PRECISIONS[args.precision]
    model, tokenizer = load(model_name)
    score, label_ids = make_scorer(model, tokenizer)
    header = header_for(model_name, examples, label_ids)
    sub = "smoke" if args.limit else "raw"
    records = run(examples, score, args.out_dir / sub / f"proofwriter_{args.precision}.jsonl", header, args.limit)
    accuracy = sum(r["correct"] for r in records) / len(records)
    print(f"{args.precision}: forced-choice accuracy {accuracy:.1%} on {len(records)}; "
          f"peak MLX memory {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
