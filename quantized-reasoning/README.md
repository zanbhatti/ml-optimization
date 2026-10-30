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

Qwen3 family, MLX on Apple Silicon. BF16 checkpoints from `mlx-community`; every lower precision is
built locally from the same BF16 weights (`mlx_lm convert`, affine quantization, group size 64), so
precision is the only difference.

| Family | Sizes | Used for |
|---|---|---|
| Qwen3-4B-Instruct-2507 | 4B | Rounds 1–2 (direct answers) |
| Qwen3 (hybrid thinking) | 0.6B, 1.7B, 4B | Rounds 3–4 (direct answers with `enable_thinking=False`; step-by-step with thinking on) |

Precisions: BF16; 8 / 6 / 5 / 4 / 3-bit round-to-nearest (RTN); and 4-bit GPTQ (`mlx_lm.quant.gptq`, same
format as RTN 4-bit) for 0.6B. GPTQ for 4B needs a block-by-block implementation (mlx-lm's needs ~19 GB).

## Benchmarks

**ProofWriter:** rule-based logical deduction with ground-truth reasoning-depth metadata (`QDep`), using
`tasksource/proofwriter`, validation split. **Planned — CLUTRR** (relational reasoning) as a second task.

## Evaluation modes and metrics

| Mode | How the answer is obtained | Runner |
|---|---|---|
| Generation | Up to 5 greedy tokens; strict parser (answer must start with the label) | `src/run_precision.py` |
| Forced choice | One forward pass; argmax over the log-probs of the True/False/Unknown tokens | `src/forced_choice.py` |
| Thinking | Greedy step-by-step reasoning (KV cache, 4,096-token budget), answer read after `</think>` | `src/thinking.py` |

Metrics: accuracy A(b, d) with Wilson 95% CIs; penalty Δ(b, d) = A(BF16, d) − A(b, d) with paired
bootstrap CIs (10,000 resamples within depth); slope of Δ on depth (the hypothesis test); per-item
margin, KL divergence and flip rate vs BF16 (forced choice); thinking length and truncations. See
[`docs/GUIDE.md`](docs/GUIDE.md) for intuition and definitions.

## Results so far

**1. Answering directly, models up to 4B can't do multi-step deduction — even at full precision.**
At every size, BF16 is ~80–90% at depth 0 (a lookup) but near chance (37–48%; chance = 33%) at
depths 1–5. So direct-answer evaluation cannot measure how quantization affects multi-step reasoning.

**2. Direct answers: 8-bit is free; 4-bit costs ~10 points.** Qwen3-4B-Instruct-2507: BF16 53.1%,
8-bit 53.9%, 4-bit 43.0% (Δ 10.2 pts, 95% CI 6.7–13.9); depth trend not significant. Answers changed
vs BF16 rise steadily as bits drop (8/6/5/4/3-bit: 2% / 4% / 12% / 22% / 49%), and the answers that
flip are the ones BF16 was unsure about — depth adds nothing once that confidence is held fixed.

**3. With step-by-step thinking, models reason — and 4-bit hurts more, especially on deep puzzles.**
Qwen3-0.6B, 180 puzzles (30 per depth):

| Depth | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| BF16 | 77 | 80 | 80 | 40 | 37 | 40 |
| 8-bit | 87 | 80 | 80 | 43 | 37 | 43 |
| 4-bit | 63 | 63 | 43 | 17 | 10 | 7 |
| 3-bit | 0 | 0 | 0 | 0 | 0 | 0 |
| 4-bit GPTQ | 83 | 70 | 57 | 27 | 20 | 20 |

- 4-bit loses 25 pts overall (95% CI 17–33), about 15 pts at depths 0–1 vs 30 at depths 2–5.
  Depth slope +3.3 pts/depth (95% CI −1.1 to +7.7): suggestive, not yet significant.
- **4-bit fails by not finishing, not by reasoning wrongly:** on the puzzles it finishes it matches
  BF16 (68% vs 66%), but it thinks ~2× longer and runs out of budget on 62/180 puzzles, rising with
  depth (3 at depth 0 → 19 at depth 5). 3-bit never finishes.
- **GPTQ halves the 4-bit loss but doesn't remove the depth pattern:** same format and size as RTN 4-bit,
  but Δ = 12.8 pts (95% CI 5.6–20.0) vs 25.0 for RTN; still ~0 at depth 0 and 13–23 pts at depths 2–5
  (slope +4.1 pts/depth, CI −0.2 to +8.3 — borderline). Out-of-budget answers drop from 62 to 39.
- **Open check:** greedy decoding encourages repetition loops; sampling and a larger budget are being
  tested before this is claimed as a quantization effect.

Full tables and charts: `analysis/` (per round and model).

## Roadmap

Phase 1 runs through December 2026 — see [`docs/PHASE1_PLAN.md`](docs/PHASE1_PLAN.md): GPTQ baseline,
speed/memory benchmark, greedy-decoding check, 1.7B and 4B thinking runs, public write-up.

## Repository Structure

```text
quantized-reasoning/
├── data/proofwriter_eval.jsonl   # Locked 540-item evaluation set (SHA-256 recorded in every result)
├── src/
│   ├── proofwriter_test.py       # Eval set, prompt, strict answer parser
│   ├── models.py                 # Model families, precision paths, on-demand quantization
│   ├── run_precision.py          # Generation runner + shared resume/ordering helpers
│   ├── forced_choice.py          # Forced-choice runner
│   ├── thinking.py               # Thinking-mode runner
│   ├── analyze*.py, size_trend.py
│   └── inference.py              # Original hand-written generation loop
├── run_*.sh                      # One-command sweeps (resumable)
├── analysis/                     # Summaries, CSVs, charts (tracked)
├── results/                      # Raw per-item records (ignored; regenerable)
├── docs/GUIDE.md, docs/PHASE1_PLAN.md
└── tests/                        # Parser, sampling, resume and statistics tests
```

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python3 -m unittest discover -s tests        # no model or download needed
./run_family_sweep.sh                        # forced choice: Qwen3 0.6B/1.7B/4B × 6 precisions
./run_thinking.sh qwen3-0.6b 180 bf16 8bit 4bit 3bit   # thinking mode, first 180 balanced items
```

All runs are resumable: rerun the same command after a stop and finished items are skipped. MLX's
buffer cache is capped (0.5 GB) and peak memory is printed at the end of each run.

## Hardware

Development and evaluation currently use a local Apple M3 Pro with 18 GB unified memory and MLX.

## Limitations

- Small-model capability and floor effects can obscure quantization penalties at difficult depths.
- Reasoning depth may correlate with input length; this potential confound requires separate analysis.
- Depth definitions are benchmark-specific.
- Quantization implementation and configuration matter, including checkpoint matching and tokenizer compatibility.
- Balanced sampling estimates performance under a balanced class distribution, not the natural dataset distribution.
- Examples can share theories; future uncertainty analysis should account for dependence.
- Thinking-mode results use greedy decoding and a 4,096-token budget; both can inflate truncation.
- 0.6B thinking results use 30 items per depth; per-depth CIs are wide (about ±15 pts).
