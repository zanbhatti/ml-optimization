"""Thinking-mode ProofWriter runs: the model reasons step by step before answering.

Same task prompts as the direct-answer runs; the chat template is rendered with
enable_thinking=True. Greedy decoding with a KV cache (mlx_lm.stream_generate).
The answer is read after </think> with the same strict normalizer as before;
outputs that hit the token budget before closing </think> are TRUNCATED and
count as incorrect. Resumable like the other runners.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from models import DEFAULT_FAMILY, FAMILIES, PRECISION_NAMES, ensure_quantized, model_path
from proofwriter_test import EVAL_PATH, INVALID, ROOT, build_or_load_eval_set, build_prompt, normalize_prediction
from run_precision import interleaved_order, item_key, load_done

THINK_END = "</think>"
TRUNCATED = "TRUNCATED"


def extract_answer(text):
    """Return (prediction, answer_text, closed_thinking)."""
    if THINK_END not in text:
        return TRUNCATED, "", False
    answer = text.split(THINK_END, 1)[1]
    return normalize_prediction(answer), answer.strip()[:200], True


def header_for(model_name, examples, max_tokens):
    return {"type": "header", "model": model_name, "mode": "thinking", "decoding": "greedy",
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": True},
            "eval_set_sha256": hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest(),
            "prompts_sha256": hashlib.sha256(json.dumps([build_prompt(e) for e in examples]).encode()).hexdigest()}


def make_generator(model, tokenizer, max_tokens):
    import mlx.core as mx
    from mlx_lm import stream_generate

    def generate(prompt):
        formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=True)
        pieces, last = [], None
        for response in stream_generate(model, tokenizer, formatted, max_tokens=max_tokens):
            pieces.append(response.text)
            last = response
        mx.clear_cache()
        return "".join(pieces), last.generation_tokens, last.finish_reason, last.generation_tps
    return generate


def record_for(index, example, text, n_tokens, finish_reason, tps, seconds):
    prediction, answer, closed = extract_answer(text)
    think_tokens_approx = len(text.split(THINK_END, 1)[0].split()) if closed else None
    return {"index": index, "key": item_key(example), "id": example["id"], "depth": example["QDep"],
            "gold": example["answer"], "prediction": prediction, "correct": prediction == example["answer"],
            "answer_text": answer, "closed_thinking": closed, "generated_tokens": n_tokens,
            "thinking_words": think_tokens_approx, "finish_reason": finish_reason,
            "tokens_per_s": round(tps, 1), "seconds": round(seconds, 2), "output": text}


def run(examples, generate, path, header, n=None, log_every=10):
    done = load_done(path, header)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = interleaved_order(examples)[:n]
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
        for k, index in enumerate(todo, 1):
            t0 = time.time()
            text, n_tokens, finish, tps = generate(build_prompt(examples[index]))
            row = record_for(index, examples[index], text, n_tokens, finish, tps, time.time() - t0)
            write(row)
            done[index] = row
            if k % log_every == 0 or k == len(todo):
                rate = (time.time() - start) / k
                print(f"  {k}/{len(todo)}  {rate:.1f}s/example  eta {rate * (len(todo) - k) / 60:.0f} min", flush=True)
    return [done[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=PRECISION_NAMES)
    parser.add_argument("--family", choices=sorted(FAMILIES), default="qwen3-4b")
    parser.add_argument("--n", type=int, help="First N items in stratum-interleaved order (balanced prefix)")
    parser.add_argument("--smoke", action="store_true", help="Write to results/thinking/smoke")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--cache-limit-gb", type=float, default=0.5)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(int(args.cache_limit_gb * 2**30))
    ensure_quantized(args.family, args.precision)
    examples = build_or_load_eval_set()
    model_name = model_path(args.family, args.precision)
    model, tokenizer = load(model_name)
    header = header_for(model_name, examples, args.max_tokens)
    sub = "smoke" if args.smoke else "raw"
    path = ROOT / "results" / "thinking" / sub / args.family / f"proofwriter_{args.precision}.jsonl"
    records = run(examples, make_generator(model, tokenizer, args.max_tokens), path, header, args.n)
    acc = sum(r["correct"] for r in records) / len(records)
    trunc = sum(r["prediction"] == TRUNCATED for r in records)
    invalid = sum(r["prediction"] == INVALID for r in records)
    mean_tokens = sum(r["generated_tokens"] for r in records) / len(records)
    print(f"{args.family} {args.precision} thinking: acc {acc:.1%} on {len(records)}; truncated {trunc}; "
          f"invalid {invalid}; mean tokens {mean_tokens:.0f}; peak MLX memory {mx.get_peak_memory() / 2**30:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
