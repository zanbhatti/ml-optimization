# Quantized Reasoning — Project Guide

A plain-language map of what this project does, why each piece exists, and what we've found.
Read top to bottom once; afterwards use it as a reference. Updated as the project changes.

---

## 1. The question, in one paragraph

A language model is billions of numbers (weights). Normally each weight is stored in 16 bits
(BF16). **Quantization** stores each weight in fewer bits (8, 6, 5, 4, 3), which shrinks memory
2–5× but rounds every weight slightly. Rounding is like adding a little blur to every calculation.
**Hypothesis:** a problem that needs 5 reasoning steps passes through that blur 5 times, so
quantization should hurt multi-step reasoning more than one-step reasoning — like a photocopy of a
photocopy. The quantity we care about is the **penalty**:

    Δ(b, d) = Accuracy(BF16, depth d) − Accuracy(b bits, depth d)

If Δ grows as depth d grows, the hypothesis holds.

---

## 2. The benchmark

**ProofWriter**: synthetic logic puzzles. Each item is a list of facts and rules plus a statement;
the model answers True, False, or Unknown.

- *Depth* (QDep) = how many rule applications are needed. Depth 0 = the answer is literally stated
  in the facts (a lookup). Depth 5 = five chained deductions.
- "Unknown" = neither the statement nor its opposite can be proved (open-world assumption).
- **Our fixed set:** 540 items = 6 depths × 3 answers × 30 each (`data/proofwriter_eval.jsonl`).
  Balanced so that always guessing one label scores exactly 33% — that is **chance**.
- The file is **locked**: every run uses the same items, and every result file stores the file's
  SHA-256 hash so we can prove it.

---

## 3. How one experiment runs, end to end

    pick model + precision ─► load weights ─► for each puzzle:
        build prompt ─► model produces an answer ─► score it ─► append one JSON line to a results file
    ─► analysis script reads all results files ─► tables + charts in analysis/

Two important properties:
- **Paired:** every precision answers the *same* 540 puzzles, so we compare item by item.
  This removes "some puzzles are just harder" noise and makes small differences detectable.
- **Resumable:** each puzzle's result is written to disk immediately (with `fsync`). If the laptop
  sleeps or crashes, rerunning the same command skips finished puzzles.

---

## 4. The three ways we ask the model

| Mode | What the model does | File | Why we use it |
|---|---|---|---|
| **Generation** (round 1) | Writes up to 5 tokens; we parse the first word | `run_precision.py` | Simplest, closest to real use |
| **Forced choice** (rounds 2–3) | One forward pass; we read the probabilities of the tokens "True", "False", "Unknown" and pick the highest | `forced_choice.py` | No formatting failures possible; gives a *confidence* number per item |
| **Thinking** (round 4) | Writes out its reasoning (hundreds–thousands of tokens) inside `<think>…</think>`, then answers | `thinking.py` | Lets the model actually do multi-step reasoning |

Why forced choice? In round 1, 4-bit sometimes ignored "answer in one word" and started explaining.
Those count as wrong, which mixes up *broke the format* with *reasoned wrongly*. Forced choice
removes the format problem entirely.

Why thinking? See §8 — without it, the models can't do multi-step reasoning at all.

---

## 5. Files

| File | What it does |
|---|---|
| `src/proofwriter_test.py` | Builds/validates the locked 540-item set, the prompt, and the strict answer parser (an answer must *start* with True/False/Unknown) |
| `src/inference.py` | Original hand-written generation loop (round 1). No KV cache: recomputes the whole prompt for every new token — fine for 5 tokens, too slow for thinking |
| `src/models.py` | Registry of model families (Qwen3 0.6B/1.7B/4B, 4B-Instruct-2507) and where each precision lives; builds quantized copies on demand with `mlx_lm convert` |
| `src/run_precision.py` | Round-1 runner (generation mode). Also holds shared helpers: balanced ordering, item hashing, resume logic |
| `src/forced_choice.py` | Forced-choice runner (rounds 2–3) |
| `src/thinking.py` | Thinking-mode runner (round 4). Uses `mlx_lm.stream_generate`, which has a KV cache (reuses past computation, so each new token is cheap) |
| `src/analyze.py` | Round-1 analysis: accuracy, Wilson CIs, bootstrap penalty CIs |
| `src/analyze_forced_choice.py` | Rounds 2–3 analysis: margins, KL, flip rates, depth slopes, confidence control |
| `src/size_trend.py` | Combines per-size analyses into one cross-size summary + chart |
| `src/analyze_thinking.py` | Round-4 analysis: accuracy and penalties by depth, depth slope, thinking length, truncations, "wrong vs didn't finish" breakdown, and a lenient-parser sensitivity check |
| `src/gptq_blockwise.py` | Block-by-block GPTQ: mlx-lm's math, one transformer block's Hessians at a time, so 1.7B/4B fit in memory |
| `src/benchmark.py` | Week 2: speed and memory per model and precision (prefill and decode tokens/s, peak memory), one subprocess per model |
| `src/analyze_benchmark.py` | Week 2: benchmark table + chart, and the bandwidth-ceiling comparison |
| `src/analyze_decoding.py` | Week 2: the greedy-decoding check (does the 4-bit penalty survive sampling and an 8k budget?) |
| `run_*.sh` | One-command sweeps (loop over precisions/sizes, then analyze). `run_week2.sh` = benchmark + decoding check |
| `tests/` | Automated checks: parser, sampling, resume, statistics on known inputs |
| `results/` | Raw per-puzzle records (not in git — large, regenerable) |
| `analysis/` | Summaries, CSVs, charts (in git) |

**Quantization details** (you should be able to say this): MLX *affine* quantization, group size 64.
Weights are split into groups of 64; each group stores a scale and offset in higher precision, and
each weight becomes a small integer (e.g. 0–15 for 4-bit). That's why "4-bit" is really ~4.5 bits
per weight. All quantized models are made from the *same* BF16 weights, so precision is the only
thing that changes.

### GPTQ — the "smart rounding" baseline

**Round-to-nearest (RTN)**, what `mlx_lm convert` does, rounds each weight to the nearest allowed
value, independently. **GPTQ** rounds the weights of a layer *one column at a time*, and after each
rounding it nudges the not-yet-rounded weights to cancel the error just introduced, so the layer's
*output* stays as close as possible to the original on real text (the "calibration data").

The math in one line: for a layer with weights W and typical inputs X, GPTQ minimizes the output
error ‖WX − ŴX‖² over rounded weights Ŵ. The matrix H = X Xᵀ (the "Hessian" of that error) says
which input directions matter most; GPTQ uses H⁻¹ to decide how to spread each rounding error onto
the remaining weights.

Our settings (`src/models.py`, precision name `gptq4`): 4-bit, group size 64 — identical format to
RTN `4bit`, so model size and speed are identical and **only the rounding method differs**. The one
layer GPTQ doesn't optimize (Qwen3's shared embedding/output layer) uses 4-bit RTN, matching `4bit`.
Calibration: mlx-lm's standard ~437 KB mixed-text file, all 512-token chunks, seed 123.

**Memory catch:** mlx-lm's GPTQ keeps one H per layer for the *whole model at once*: ~2.0 GB for
0.6B, ~6.6 GB for 1.7B, **~19.3 GB for 4B** (doesn't fit in 18 GB). `ensure_quantized` refuses to
start above an 8 GB budget. The 4B needs a block-by-block GPTQ (one transformer block's Hessians at a
time, ~0.5 GB) — planned.

**Why it matters:** if 4-bit damage mostly disappears under GPTQ, the finding is about naive rounding,
not about 4-bit itself. Reviewers and interviewers will ask.

#### A toy example: why compensating beats rounding

A layer with two weights, both 1.4, allowed values are whole numbers. Its inputs are strongly
correlated: in practice x₁ ≈ x₂, say both 1. True output: 1.4 + 1.4 = **2.8**.

- **Round-to-nearest:** 1.4 → 1 and 1.4 → 1. Output 2.0. **Error 0.8.** Each weight was rounded
  "well" on its own, but both errors point the same way and add up.
- **GPTQ:** round the first weight 1.4 → 1 (error −0.4). Because x₂ ≈ x₁, the second weight can make
  up for it: nudge it to 1.4 + 0.4 = 1.8, then round → 2. Output 1 + 2 = 3.0. **Error 0.2.**

GPTQ made one weight *worse* on its own (1.4 → 2 is further than 1.4 → 1) to make the layer's
*output* better. The Hessian H = XXᵀ is what tells it which weights can cover for which: here,
x₁ and x₂ move together, so w₂ can absorb w₁'s error. With uncorrelated inputs there's nothing to
exploit and GPTQ reduces to plain rounding.

#### Block-by-block GPTQ — our own implementation (`src/gptq_blockwise.py`, precision `gptq4b`)

**Problem:** mlx-lm's GPTQ holds every layer's H at once (19.3 GB for 4B). Building the 1.7B model
hit 13.3 GB and was stopped by our 12 GB safety rule.

**Idea:** you don't need all the H's at once, only the current layer's. So walk through the model
one transformer block at a time:
1. Keep the calibration text's hidden states X at the input of the current block (~0.2–0.5 GB).
2. Run just that block on X and collect its H's. Layers that read the *same* input have the
   *same* H, so q/k/v share one, gate/up share one, and o_proj and down_proj get their own: 4
   per block instead of 7.
3. Quantize the block's layers with exactly mlx-lm's column-by-column math (1% damping,
   Cholesky inverse, group-wise scales).
4. The block's output, computed with its **original** weights, becomes the next block's X. That's
   what mlx-lm does too (all its H's come from one pass through the full-precision model), so
   the result should be the same.
5. Free the H's and move on.

Extra memory beyond the weights: ~0.4 GB (0.6B), ~1.0 GB (1.7B), ~1.4 GB (4B).

**How we know it's right:**
- **Unit tests** (`tests/test_gptq_blockwise.py`): on tiny 2-block Qwen3 models, every quantized
  integer, scale and bias is *bit-identical* to mlx-lm's output, with tied and untied output layers.
  The tests caught one real bug: mlx-lm adds damping to H *in place*. Once q/k/v share one H, the
  first layer's damping leaked into the next two. Fixed by copying H before damping.
- **Validation on 0.6B** against the existing mlx-lm `gptq4` model (`src/validate_gptq.py`,
  `run_gptq_validation.sh`, output in `analysis/gptq_validation/qwen3-0.6b/validation.json`):
  - All 196 GPTQ layers have *identical* integers (100%, max level difference 0), and the per-layer
    output error is identical too (ratio 1.000, mean relative error 0.0032 for both).
  - Forced choice: the same 540 predictions, identical log-probs, KL 0. Thinking: all 180 outputs are
    identical token for token (46.1%).
  - Peak memory is 1.30 GB MLX (1.9 GB process), against mlx-lm's ~2 GB of Hessians alone. The build
    took ~3.2 min.
  - So it is the same algorithm, just with less memory. Building 1.7B/4B still waits until that's
    reviewed.

**`--sequential` (not used yet):** the GPTQ paper feeds each block the output of the
*already-quantized* previous blocks, so later layers learn to correct earlier rounding. mlx-lm
doesn't do this. The flag is there for a possible future experiment.

### Speed and memory — why anyone quantizes (week 2)

Accuracy loss is only half the trade. The other half is what you get for it: a smaller model that
fits in less memory and generates faster. `src/benchmark.py` measures that.

**Two phases of running a model:**
- **Prefill** = reading the prompt. All L prompt tokens go through the model *together*, so each
  weight, once fetched from memory, is used L times.
- **Decode** = generating the answer, *one token at a time*. Each new token needs every weight fetched
  again, and uses each one for just ~2 arithmetic operations (a multiply and an add).

**The key idea, arithmetic intensity** = arithmetic operations per byte fetched from memory. The
GPU can do far more arithmetic per second than it can fetch bytes (M3 Pro: ~150 GB/s of memory
bandwidth). Decode does ~2 operations per weight, so the GPU sits idle waiting for weights:
**decode is memory-bandwidth-bound.** Prefill reuses each weight L times, so it's **compute-bound**.

**What follows (the testable prediction):**
- Decode speed ≤ bandwidth ÷ bytes per token ≈ 150 GB/s ÷ model size. A 4-bit model is ~3.5×
  smaller than BF16, so it should decode up to ~3.5× faster.
- Prefill should barely change with precision; quantized weights must be unpacked before use, so it
  can even be slightly slower.

**Results (`analysis/benchmark/`, 3 repeats, 512-token context unless noted):**

| | BF16 decode | 4-bit decode | Speedup | BF16 % of ceiling | Prefill BF16 → 4-bit | Peak memory @4k, BF16 → 4-bit |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 80 tok/s | 184 tok/s | 2.3× | 63% | 3002 → 3117 tok/s | 2.4 → 1.7 GB |
| Qwen3-1.7B | 31 | 112 | 3.7× | 70% | 1127 → 1358 | 4.5 → 2.2 GB |
| Qwen3-4B | 16 | 52 | 3.2× | 87% | 525 → 530 | 9.3 → 3.7 GB |

- **The prediction holds.** Decode speed tracks weight size along the bandwidth ceiling (chart, left
  panel). The 4B in BF16 runs at 87% of the theoretical maximum: it is almost purely waiting on memory.
- **Prefill doesn't care about precision** (right panel): same speed within noise at every bit
  width, because it's limited by arithmetic, not memory.
- **Small models gain less.** The 0.6B at 4-bit reaches only 41% of its ceiling. Its weights
  (0.34 GB) are so small that fixed per-token costs dominate: launching GPU work, reading the KV
  cache, sampling. Shrinking weights further can't remove those.
- **GPTQ costs nothing at inference:** identical speed and memory to RTN 4-bit (same format).
- **Why thinking mode makes this matter:** a 1,700-token reasoning trace is almost all decode. A
  4-bit 4B "thinks" ~3× faster than BF16, which is the practical reason people want 4-bit to work for
  reasoning.

**How it's measured:** each model is loaded in its own process (so peak memory is that model's
alone) and gets a warm-up run. Contexts are 512 and 4,096 tokens of real ProofWriter text. 128
tokens are decoded with the end-of-text token blocked, so every run does the same work. Each
measurement is repeated 3× and the median kept. **Caveat:** laptops slow down when hot (thermal
throttling), so absolute numbers are approximate; ratios between precisions are more reliable.

### The greedy-decoding check (week 2)

**Worry:** 4-bit's thinking-mode loss is mostly "never finished" — it hits the 4,096-token budget.
Two boring explanations could produce that without compression truly breaking reasoning:
1. **Greedy decoding loops.** Always taking the single most likely token can lock a model into
   repeating itself. Qwen recommends *sampling* for thinking (temperature 0.6, top-p 0.95, top-k 20:
   pick randomly among the plausible next tokens). Maybe 4-bit only loops under greedy.
2. **The budget is just too small.** Maybe 4-bit is wordier but would finish with more room.

**Test (`run_week2.sh`, 0.6B):**
1. Rerun BF16 and 4-bit with Qwen's sampling settings, 2 random seeds each, all 180 items. Each item
   is seeded by (seed, item), so a resumed run reproduces the same outputs.
2. Rerun only the items that hit 4,096 tokens, now with 8,192, greedy. Greedy is deterministic, so
   every item that already finished would come out identical; no need to redo them.

**Reading it:** if Δ (BF16 − 4-bit) and its growth with depth survive both, the effect is
compression, not a decoding artifact. If the gap closes, the honest claim becomes "4-bit makes the
model loop under greedy decoding." That's still a finding, but a different one.

*Early signal from the smoke test:* of 8 truncated 4-bit items given 8,192 tokens, 7 still ran out.
Some were literally repeating one sentence ("But that would mean they are both furry and green…").
More room doesn't seem to rescue a looping model. The full run will tell.

---

## 6. Metrics — intuition first

**Accuracy** — fraction correct. Simple, but coarse: an item only counts when the answer flips.

**Margin** (forced choice) — how confidently the model picks the right answer:

    margin = log p(gold label) − max log p(any wrong label)

Positive = right, and bigger = more confident. Measured in *nats* (natural-log units; a margin of
2.3 nats ≈ the right answer is 10× more likely than the best wrong one). Why: accuracy only notices
when a margin crosses zero; margin notices damage *before* answers flip → much more statistical
power from the same 540 items.

**Margin loss** ΔM = margin(BF16) − margin(quantized), per item. Positive = quantization hurt.

**Flip rate** — % of items where the quantized model's answer differs from BF16's. Captures damage
even when accuracy doesn't move (a model near chance can flip many answers without changing its score).

**KL divergence** — how different the quantized model's probabilities over {True, False, Unknown}
are from BF16's. Zero = identical beliefs.

**Prediction share** — how often each label is predicted. Gold is 33/33/33, so big deviations mean
the model has a *label bias* (e.g. says "Unknown" 95% of the time).

---

## 7. Statistics — what and why

**Wilson 95% interval** (accuracy). With 90 items per depth, a measured 50% could really be ~40–60%.
Wilson is the standard interval for a proportion that behaves well near 0% and 100%.

**Paired bootstrap** (penalties). To get a confidence interval for Δ without assuming a formula:
resample the 90 items of each depth *with replacement* 10,000 times, recompute Δ each time, and take
the middle 95% of those values. "Paired" = each resampled item brings its BF16 *and* quantized result
together, so item difficulty cancels out. "Within depth" = each depth keeps exactly 90 items.

**Depth slope** — the core hypothesis test. Fit a straight line through Δ vs depth; the slope is
"extra penalty per additional reasoning step". Slope > 0 with a CI above zero supports the
hypothesis. Bootstrap gives its CI.

**Confidence control** — a check against a confound. Deep items are ones the BF16 model is already
*unsure* about, and unsure answers are easy to tip over. So "deep items flip more" could just mean
"unsure items flip more". We fit:

    flip ≈ a + b·(BF16 confidence gap) + c·(item is deep)

and ask whether c (the depth effect *holding confidence fixed*) is above zero. Confidence gap =
BF16's top label probability minus its second, in log units.

---

## 8. Results so far

**Round 1 — generation, Qwen3-4B-Instruct-2507, BF16/8/4-bit.**
8-bit matched BF16 (−0.7 pts). 4-bit lost 10.2 pts overall (95% CI 6.7–13.9), from 10 pts at
depth 0 to 16 at depth 5, but the depth trend was not significant. About half the 4-bit loss was
format failures (it started explaining instead of answering).

**Round 2 — forced choice, same model, BF16 + 8/6/5/4/3-bit.**
Flip rate rises steadily as bits drop: 2% → 4% → 12% → 22% → 49%. The answers that flip are the
ones BF16 was unsure about; once confidence is held fixed, depth adds nothing. 3-bit nearly stops
saying "Unknown".

**Round 3 — forced choice, Qwen3 0.6B / 1.7B / 4B × 6 precisions.**
At every size, BF16 is ~82–90% at depth 0 but **near chance (37–48%) at depths 1–5**. Small models
under heavy quantization mostly shift their label bias (1.7B at 4-bit says "Unknown" 95% of the time).
Depth never predicted flips in any of the 20 model × precision combinations.

**The key insight from rounds 1–3:** answering in one word gives the model exactly *one* pass
through its layers, no matter how many steps the puzzle needs. These models can do lookups that way
but not deductions — so there was no reasoning chain for quantization to damage. The depth question
was untestable in that mode.

**Round 4 — thinking mode.**
*Smoke test (18 items):* Qwen3-4B BF16 got depths 0–3 all right (12/12) and struggled at 4–5 (2/6).
Cost: ~134 s per puzzle at BF16 (16 tokens/s), ~44 s at 4-bit, ~13 s for 0.6B.

*Qwen3-0.6B, 180 items (30 per depth), BF16 / 8 / 4 / 3-bit* (`analysis/thinking/qwen3-0.6b/`):
- BF16 now reasons: ~80% at depths 0–2, ~40% at 3–5 — real headroom (vs near-chance beyond depth 0 when answering directly).
- 8-bit is still free (Δ −2.8 pts, CI −9.4 to +3.9).
- 4-bit loses 25 pts overall (CI 17–33) — 2.5× its direct-answer loss — and more on deep puzzles:
  ~15 pts at depths 0–1 vs ~30 at depths 2–5. Depth slope +3.3 pts/depth, CI −1.1 to +7.7:
  suggestive, **not yet significant** (lenient parser: +4.1, CI −0.1 to +8.3).
- **How 4-bit fails:** when it finishes, it is as accurate as BF16 on the same items (68% vs 66%).
  It fails by *not finishing*: it thinks ~2× longer at every depth and runs past the 4,096-token
  budget on 62/180 items, rising with depth (3 at depth 0 → 19 at depth 5); some are literal loops.
- 3-bit collapses completely: every answer runs out of budget.
- **GPTQ 4-bit** (same size/format as RTN 4-bit): loses 12.8 pts (CI 5.6–20.0), half of RTN's 25.
  The depth pattern remains: ~0 at depth 0, 13–23 pts at depths 2–5; slope +4.1 pts/depth
  (CI −0.2 to +8.3, borderline). Out-of-budget answers 39 vs 62. Direct answers agree: flip rate
  28% vs 61%, and no collapse toward one label.

*Caveats checked in week 2 (below):* greedy decoding and the token budget.

### Round 5 — the greedy-decoding check (week 2, 0.6B)

Files: `analysis/thinking/qwen3-0.6b/decoding_check.{md,png}`. Lenient parser, because sampled 4-bit
often writes its answer as prose ("The statement is **true**"), which the strict parser rejects; that
would inflate the strict penalty to 36 points for a formatting reason.

| Setup | 4-bit Δ vs BF16 | Depth slope | 4-bit truncated | Accuracy on answered, BF16 vs 4-bit |
|---|---|---|---|---|
| Greedy, 4,096 tokens | 18.3 [10.0, 26.7] | +4.1 [−0.1, +8.3] | 34% | 59% vs 62% |
| Greedy, 8,192 tokens | 17.8 [9.4, 26.1] | +3.1 [−1.2, +7.2] | 30% | — |
| Sampled, 2 seeds | 17.8 [11.1, 24.4] | −1.1 [−4.9, +2.6] | 16% | 66% vs 57% |

**What it means, in plain words:**
1. **More room doesn't help.** Given twice the tokens, 54 of 62 stuck 4-bit outputs still ran out.
   They weren't slow, they were looping (repeating the same sentence). GPTQ: 39 of 39 still ran out.
2. **Sampling breaks most loops,** halving truncations, but 4-bit still loses ~18 points. So the
   overall loss is not a greedy artifact; it's robust.
3. **The failure mode changes.** Under greedy, 4-bit fails by *looping* and is fine when it
   finishes. Under sampling, it loops less but makes more *wrong* conclusions when it finishes
   (57% vs 66%). The damage is the same size; it just shows up differently. One interpretation: the
   quantized model's confidence in each next step is flatter. Greedy turns that into getting stuck;
   sampling turns it into wandering off.
4. **The depth effect does not survive.** Under sampling, Δ is flat across depth (slope ≈ −1). The
   "deeper puzzles suffer more" pattern was tied to greedy looping, which builds up over longer chains.

**Revised claim:** at 0.6B, 4-bit RTN costs ~18 points of reasoning accuracy at every depth, robust to
decoding and budget. The depth-dependent part is a greedy-decoding phenomenon.

**Why not give up on the depth hypothesis yet:** BF16 0.6B is near chance at depths 3–5 (38–48%),
which squeezes how big Δ can be on deep puzzles. 1.7B (week 3) has more headroom and is the real
test. Week 3 should use sampling, since that is how these models are meant to run.

---

## 9. Decisions log

| Decision | Why |
|---|---|
| Lock the 540-item set and hash it | Same items every run; proves nothing changed |
| Build quantized models locally from one BF16 checkpoint | Guarantees only precision differs (same weights, tokenizer, template) |
| Greedy decoding (always pick the top token) | Deterministic: rerunning gives identical outputs, so differences come from precision, not randomness |
| Cap MLX memory cache (0.5 GB) and clear it per puzzle | The first long run filled RAM and swapped; MLX's cache is unbounded by default |
| Forced choice instead of parsing text | Removes format failures; gives continuous confidence |
| Switch from 4B-Instruct-2507 to original Qwen3 family | 2507 exists only at 4B; original Qwen3 has 0.6B→32B with one architecture, and a thinking on/off switch in the same weights |
| `enable_thinking=False` for direct answers | Original Qwen3 models think by default; the switch makes them answer directly |
| Thinking budget 4,096 tokens; over-budget = TRUNCATED = wrong | Bounds runtime; truncation rate is reported so we can see if the budget is too small |
| Report a lenient parser alongside the strict one (thinking mode) | 4-bit often ends with "Answer: Unknown", which the strict (pre-registered) parser rejects; showing both keeps the primary metric honest without hiding a scoring artifact |
| Benchmark: one subprocess per model, EOS blocked, median of 3 | Clean per-model peak memory; every run does identical work; medians resist one-off hiccups |
| Decoding check: Qwen's own sampling settings, 2 seeds | The "fairest" non-greedy setup is the one the model's makers recommend; 2 seeds shows how much randomness moves the result |
| 8k budget: rerun only truncated items | Greedy is deterministic, so finished items can't change; halves the compute |
| Decoding check scored with the lenient parser | Sampled 4-bit states answers in prose ("is **true**"); strict scoring would count format drift as reasoning failure |
| Week 3+ runs use sampling (2 seeds) as the main setup | It's how Qwen3 is meant to run for thinking, and the week-2 check showed greedy changes the failure mode |
| Balanced ("interleaved") item order | Any first-N subset has equal items per depth and label, so partial runs are still fair comparisons |

---

## 10. Can you answer these? (self-check)

1. Why does quantization make a model smaller, and why is "4-bit" ~4.5 bits per weight?
2. Why is chance 33% on our set, and why does that matter for interpreting depths 1–5?
3. What does a margin of +2.3 nats mean in plain words?
4. Why can 3-bit change half its answers yet keep the same accuracy as BF16?
5. What would the depth slope look like if the hypothesis were true?
6. Why do we need the confidence control — what alternative explanation does it rule out?
7. Why did one-word answers make the depth question untestable, and why does thinking fix that?
8. Why must every precision answer the *same* items?
9. In thinking mode, 4-bit is as accurate as BF16 on the items it finishes. Why does that change how we describe what compression breaks?
10. Why could greedy decoding make the 4-bit result look worse than it really is?
11. What does GPTQ do that round-to-nearest doesn't, and why does it need calibration data?
12. Why does mlx-lm's GPTQ need ~19 GB for the 4B model — what is being stored?
13. Why is decoding limited by memory bandwidth but prefill limited by compute? (Hint: how many
    times is each weight used per fetch?)
14. 4-bit is 3.5× smaller than BF16. Why does the 4B decode ~3.2× faster but the 0.6B only ~2×?
15. What result from the decoding check would make you drop the claim "4-bit hurts deep reasoning"?
16. In the two-weight toy example, why does GPTQ deliberately round one weight the "wrong" way?
17. Why can q_proj, k_proj and v_proj share one Hessian, and why does block-by-block GPTQ give the
    same answer as doing the whole model at once?
