# Quantized Reasoning

An ongoing study of how post-training quantization affects language-model reasoning as required reasoning depth increases. Local inference uses MLX on Apple Silicon.

## Research Question

Does post-training quantization disproportionately degrade reasoning performance as reasoning depth increases?

## Hypothesis

We hypothesize that the quantization penalty

Δ(b, d, t) = A(16, d, t) − A(b, d, t)

increases with reasoning depth. Here, A is classification accuracy, b is numerical precision in bits, d is benchmark-defined reasoning depth, and t is task family. A(16, d, t) is the BF16 baseline. A decline in baseline accuracy with depth alone does not establish this hypothesis; the gap between precision levels must be measured.

## Motivation

Lower precision reduces model memory requirements and can reduce computational requirements, but perturbs numerical representations. We investigate whether those perturbations disproportionately affect multi-step reasoning. No causal mechanism has been established by this project.

## Experimental Design

- Frozen pretrained models; no fine-tuning.
- Identical examples, prompts, greedy decoding, normalization, and scoring across precision levels.
- ProofWriter validation data stratified by depth 0–5 and answer class (True, False, Unknown).
- Exactly 30 examples per class at each depth: 90 per depth, 540 total.
- Seed 42 with stable candidate ordering and sampling without replacement within each stratum; identical ID/theory/question/answer/depth rows are deduplicated across source configurations before sampling.
- A persisted evaluation file is loaded on subsequent runs; invalid or incomplete files cause an error instead of resampling.
- The existing open-world prompt is preserved: False requires deriving the opposite statement; inability to prove either side means Unknown.

Commit and reuse `data/proofwriter_eval.jsonl` for every precision run. The persisted content, rather than the seed alone, controls for upstream dataset changes. Each row includes ID, theory, question, answer, QDep, dataset name, split, and sampling seed. IDs, theories, and questions together identify evaluation items.

Generation uses the tokenizer's chat template, greedy argmax, EOS stopping, and a maximum of five new tokens. Matching checkpoints must use the same tokenizer/chat template and evaluation code; changing precision alone is the intended intervention. Saved reports include evaluation-file and prompt hashes, model name, decoding settings, metrics, and raw predictions. The prompt hash covers the task prompts before chat-template formatting; tokenizer/template compatibility must also be checked when adding quantized checkpoints.

## Models

Current: Qwen3 4B Instruct BF16 (`mlx-community/Qwen3-4B-Instruct-2507-bf16`).

Planned: matching Qwen3 4B Instruct BF16, 8-bit, and 4-bit MLX checkpoints. The full precision-comparison runner is not implemented.

## Benchmarks

**Current — ProofWriter:** rule-based logical deduction with ground-truth reasoning-depth metadata (`QDep`), using `tasksource/proofwriter`, validation split.

**Planned — CLUTRR:** relational reasoning, where chain length provides another notion of depth. Depths across task families will be analyzed separately rather than assumed equivalent.

## Metrics

A(b, d, t) is correct predictions divided by all evaluated examples for precision b, depth d, and task family t. The primary metric is accuracy by reasoning depth; the primary comparison is Δ(b, d, t), defined above.

The evaluator reports correct/total, invalid prediction counts, per-depth accuracy, and overall accuracy. A prediction must begin, after whitespace, with the case-sensitive label True, False, or Unknown, followed by a supported punctuation delimiter, whitespace, or end of output. The first label is used. Other outputs are INVALID and count as incorrect. Per-example gold and predicted labels are retained for future per-label accuracy and confusion analysis.

Uncertainty estimates and statistical analysis of depth-dependent quantization penalties are planned, not implemented.

## Current Status

Work in progress.

Completed:

- Custom MLX autoregressive inference with tokenizer-provided EOS handling.
- ProofWriter loading and open-world True/False/Unknown evaluation.
- Depth-based evaluation and label stratification.
- Initial BF16 pilot experiments (15 examples per depth); these are not final findings.
- Identification and control of early experimental confounds through shared evaluation conditions.
- Reproducible sampling and persisted evaluation-set support, with validation and lightweight tests.
- Fixed 540-example ProofWriter evaluation artifact created, validated, and locked for version control (`data/proofwriter_eval.jsonl`).

In progress:

- Establishing a larger BF16 baseline; the 540-example model evaluation has not been run.
- Quantized model evaluation.

## Roadmap

1. **Completed:** Lock the reproducible 540-example ProofWriter evaluation set.
2. Establish the larger BF16 baseline.
3. Evaluate matching 8-bit and 4-bit models.
4. Compute quantization penalties by depth.
5. Add uncertainty estimates and statistical analysis.
6. Investigate input-length and class-specific effects.
7. Add CLUTRR.
8. Optionally collect systems metrics, including memory usage and inference latency.
9. Analyze and document results.

## Repository Structure

```text
quantized-reasoning/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   └── proofwriter_eval.jsonl # Locked 540-example evaluation set
├── src/
│   ├── inference.py          # Model loading configuration and autoregressive inference
│   └── proofwriter_test.py   # Sampling, persistence, prompts, evaluation, and CLI
└── tests/
    └── test_proofwriter.py   # CPU-only sampling, normalization, and scoring tests
```

Generated artifacts: `data/proofwriter_eval.jsonl` (created by preparation; intended for version control) and `results/proofwriter_bf16.json` (created by evaluation; ignored by Git).

## Running

From the repository root, on Apple Silicon:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Prepare or validate the evaluation set without loading a model:

```bash
python src/proofwriter_test.py --prepare-only
```

Run the BF16 evaluation:

```bash
python src/proofwriter_test.py
```

The first run needs access to Hugging Face to download uncached dataset/model files. Preparation requires only `datasets`; inference also requires MLX-LM. Paths for generated artifacts are resolved relative to the repository, independently of the working directory. Keep the evaluation JSONL intact for subsequent runs.

Progress is printed every 15 examples, followed by per-depth and overall summaries. Use `--debug` for individual predictions, or `--output results/my_run.json` to choose a report path. The default report is overwritten on reruns; use distinct paths to retain separate runs.

Run the lightweight tests without MLX or dataset downloads:

```bash
python3 -m unittest discover -s tests
```

## Hardware

Development and evaluation currently use a local Apple M3 Pro with 18 GB unified memory and MLX.

## Limitations

- Small-model capability and floor effects can obscure quantization penalties at difficult depths.
- Reasoning depth may correlate with input length; this potential confound requires separate analysis.
- Depth definitions are benchmark-specific.
- Quantization implementation and configuration matter, including checkpoint matching and tokenizer compatibility.
- Balanced sampling estimates performance under a balanced class distribution, not the natural dataset distribution.
- Examples can share theories; future uncertainty analysis should account for dependence.
- Pilot results are not final experimental findings and do not establish a quantization effect.
