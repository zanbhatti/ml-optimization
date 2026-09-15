"""Controlled ProofWriter evaluation; model imports occur only for inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

DATASET_NAME = "tasksource/proofwriter"
SEED = 42
DEPTHS = [0, 1, 2, 3, 4, 5]
LABELS = ["True", "False", "Unknown"]
EXAMPLES_PER_LABEL = 30
MAX_NEW_TOKENS = 5
INVALID = "INVALID"
ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "proofwriter_eval.jsonl"
FIELDS = ("id", "theory", "question", "answer", "QDep")


def normalize_prediction(output):
    """Accept a leading classification, never a label inside an explanation."""
    match = re.match(r"^\s*(True|False|Unknown)(?=$|\s|[.,!?:;])", output)
    return match.group(1) if match else INVALID


def build_prompt(example):
    return (
        "Use only the facts and rules below.\n\n"
        "Classify the statement as:\n"
        "- True: the statement can be derived from the facts and rules.\n"
        "- False: the opposite of the statement can be derived from the facts and rules.\n"
        "- Unknown: neither the statement nor its opposite can be derived.\n\n"
        f"Facts and rules:\n{example['theory']}\n\n"
        f"Statement:\n{example['question']}\n\n"
        "Answer with exactly one word: True, False, or Unknown."
    )


def validate_eval_set(examples):
    counts = Counter()
    identities = set()
    for example in examples:
        if any(field not in example for field in FIELDS):
            raise ValueError("Evaluation row is missing required fields")
        if not all(isinstance(example[key], str) for key in ("theory", "question", "answer")):
            raise ValueError("Theory, question and answer must be strings")
        if type(example["QDep"]) is not int:
            raise ValueError("QDep must be an integer")
        # Some dataset versions use theory IDs shared across questions.
        identity = (example["id"], example["theory"], example["question"])
        if identity in identities:
            raise ValueError(f"Duplicate evaluation question: {identity}")
        identities.add(identity)
        counts[example["QDep"], example["answer"]] += 1
    expected = Counter({(d, label): EXAMPLES_PER_LABEL for d in DEPTHS for label in LABELS})
    if counts != expected:
        raise ValueError(f"Expected exactly 30 examples in every depth/label stratum; got {dict(counts)}")


def build_or_load_eval_set(path=EVAL_PATH, dataset=None):
    path = Path(path)
    if path.exists():
        examples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        validate_eval_set(examples)
        return examples
    if dataset is None:
        from datasets import load_dataset
        dataset = load_dataset(DATASET_NAME, split="validation")
    buckets = {(d, label): [] for d in DEPTHS for label in LABELS}
    for row in dataset:
        key = (row["QDep"], row["answer"])
        if key in buckets:
            buckets[key].append({field: row[field] for field in FIELDS})
    rng = random.Random(SEED)
    examples = []
    for key, candidates in buckets.items():
        # Stable ordering makes selection independent of dataset iteration order.
        unique = {json.dumps(row, sort_keys=True): row for row in candidates}
        candidates = [unique[key] for key in sorted(unique)]
        if len(candidates) < EXAMPLES_PER_LABEL:
            raise ValueError(f"Stratum {key} has {len(candidates)} rows; need {EXAMPLES_PER_LABEL}")
        for row in rng.sample(candidates, EXAMPLES_PER_LABEL):
            examples.append(dict(row, sampling_seed=SEED, dataset=DATASET_NAME, split="validation"))
    validate_eval_set(examples)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing a previously locked evaluation set.
    with path.open("x") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    return examples


def summarize(records):
    total = len(records)
    correct = sum(row["correct"] for row in records)
    return {"correct": correct, "total": total,
            "invalid": sum(row["prediction"] == INVALID for row in records),
            "accuracy": correct / total if total else None}


def evaluate_depth(examples, predict, debug=False):
    records = []
    for example in examples:
        raw = predict(build_prompt(example))
        prediction = normalize_prediction(raw)
        record = {"id": example["id"], "question": example["question"],
                  "depth": example["QDep"], "gold": example["answer"],
                  "raw_prediction": raw, "prediction": prediction,
                  "correct": prediction == example["answer"]}
        records.append(record)
        if debug:
            print(json.dumps(record))
        elif len(records) % 15 == 0:
            print(f"  {len(records)}/{len(examples)} examples", flush=True)
    return summarize(records), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Persist/validate examples without loading a model")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "proofwriter_bf16.json")
    args = parser.parse_args()
    examples = build_or_load_eval_set()
    print(f"Validated {len(examples)} examples: {EVAL_PATH}", flush=True)
    if args.prepare_only:
        return
    from inference import MODEL_NAME, run_inference
    from mlx_lm import load
    model, tokenizer = load(MODEL_NAME)
    def predict(prompt):
        return run_inference(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS)
    records, by_depth = [], {}
    for depth in DEPTHS:
        print(f"Depth {depth}", flush=True)
        metrics, rows = evaluate_depth([e for e in examples if e["QDep"] == depth], predict, args.debug)
        by_depth[depth] = metrics
        records.extend(rows)
        print(f"  Accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.1%}; invalid: {metrics['invalid']}", flush=True)
    overall = summarize(records)
    print(f"Overall: {overall['correct']}/{overall['total']} = {overall['accuracy']:.1%}; invalid: {overall['invalid']}")
    report = {"model": MODEL_NAME, "eval_set_sha256": hashlib.sha256(EVAL_PATH.read_bytes()).hexdigest(),
              "decoding": {"method": "greedy", "max_new_tokens": MAX_NEW_TOKENS},
              "prompts_sha256": hashlib.sha256(json.dumps([build_prompt(e) for e in examples]).encode()).hexdigest(),
              "by_depth": by_depth, "overall": overall, "predictions": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()
