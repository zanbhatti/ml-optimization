# Phase 1 plan — Quantized Reasoning (Nov–Dec 2026)

**Constraints:** M3 Pro 18 GB Mac only (no PACE). ~4 hrs/week of Zan's time; compute runs unattended
(overnight or during class), resumable, memory-capped. Phase 1 done by Dec 31 so Phases 2–3 finish
before Fall 2027 recruiting.

**Scope**
- Keep: thinking-mode runs at Qwen3 0.6B / 1.7B / 4B; thinking vs direct answers on the same weights.
- Add: a real quantization baseline (**GPTQ**, via `mlx_lm.quant.gptq`, runs locally) next to MLX
  round-to-nearest (RTN); a speed/memory benchmark per precision; a check of whether greedy decoding
  inflates 4-bit's "fails to finish" effect.
- Optional: Qwen3-8B with **8-bit as the reference** (BF16 8B ≈ 16 GB does not fit in 18 GB) —
  stated plainly in the report.
- Cut for now: 14B/32B, the vLLM/PyTorch port, a workshop paper as a requirement (only if results
  are clean by late January).

**Why GPTQ matters:** RTN rounds each weight independently. GPTQ chooses rounded values that
minimize the error of each layer's *output* on calibration text, compensating for earlier rounding
errors. It's the standard baseline reviewers expect; if RTN's 4-bit damage disappears under GPTQ,
the finding is about a naive method, not about 4-bit itself. `mlx_lm` GPTQ writes the same affine
format (same bits, same group size), so speed is identical and only the rounding method differs.

## Week by week

| Week | Compute (unattended) | Zan (~4 hrs) |
|---|---|---|
| **Oct 26 – Nov 1** | Nothing new (PSYC exam 10/28, AMEX 10/29) | — |
| **1: Nov 2 – 8** | Download GPTQ calibration file (~1 MB text). Build GPTQ 4-bit for 0.6B and 4B (~minutes each). Direct-answer forced-choice on GPTQ models (~10 min). 0.6B thinking, GPTQ 4-bit, 180 items (~70 min). | Read GUIDE §5–7 + the new GPTQ section; explain RTN vs GPTQ in your own words |
| **2: Nov 9 – 15** | Speed/memory benchmark: prefill and decode tokens/s, latency, peak memory for 0.6B/1.7B/4B × BF16/8/4/3/GPTQ (~45 min). Greedy check on 0.6B BF16 vs 4-bit: sampling (temp 0.6, top-p 0.95, 2 seeds) and an 8,192-token budget on truncated items (~3 hrs, overnight). | Review benchmark; learn why decoding is memory-bandwidth-bound (the "arithmetic intensity" idea); decide whether the 4-bit derailment claim survives |
| **3: Nov 16 – 22** | 1.7B thinking runs: BF16 / 8 / 4 RTN / 4 GPTQ, 180 items (overnight, ~4 hrs) | Revise the public write-up draft (I draft, you edit); README cleanup |
| **4: Nov 23 – 29** (Thanksgiving) | Run configs saved as files (reproducibility); repo polish | **Publish write-up v1** (blog post / README results) |
| **5: Nov 30 – Dec 6** | 4B thinking: BF16 vs 4-bit RTN vs 4-bit GPTQ, 90 items, in 2–3 overnight chunks (~6 hrs total) | Read results; self-check questions |
| **6: Dec 7 – 13** (finals) | Optional: 8B direct-answer with 8-bit reference | Light week |
| **7: Dec 14 – 20** | Final analysis across sizes and methods | **Independent review:** explain the math, justify decisions, reproduce one key number yourself |
| **8: Dec 21 – 31** | — | Write-up v2 (final), merge to main, update resume bullet |

## What "done" means
- A write-up a hiring manager can read in 10 minutes, with numbers, charts and honest caveats.
- Repo at a research-engineering bar: tests, saved run configs, a clean README, resumable runs,
  `docs/GUIDE.md` current.
- Zan can explain every design decision and the statistics without notes.
