# Qwen3 size trend — forced-choice ProofWriter (540 paired items per model)

## BF16 baseline: is there headroom beyond depth 0?

| Model | Acc depth 0 | Mean acc depths 1–5 | (chance 33%) |
|---|---|---|---|
| qwen3-0.6b | 57.8% | 43.6% | |
| qwen3-1.7b | 83.3% | 36.9% | |
| qwen3-4b | 82.2% | 48.2% | |

## Answers changed vs BF16 (flip rate)

| Model | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| qwen3-0.6b | 7.8% | 11.9% | 20.9% | 60.9% | 56.1% |
| qwen3-1.7b | 3.1% | 19.8% | 17.2% | 15.7% | 80.6% |
| qwen3-4b | 3.5% | 8.0% | 8.7% | 28.0% | 31.9% |

## Margin loss vs BF16 (nats, [95% CI])

| Model | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| qwen3-0.6b | -0.02 [-0.04, +0.00] | +0.07 [+0.03, +0.11] | +0.20 [+0.13, +0.27] | +0.93 [+0.76, +1.11] | +0.55 [+0.42, +0.67] |
| qwen3-1.7b | -0.02 [-0.11, +0.07] | -0.43 [-0.61, -0.24] | +0.08 [-0.18, +0.34] | +4.72 [+3.58, +5.84] | +2.12 [+1.45, +2.80] |
| qwen3-4b | -0.09 [-0.14, -0.03] | -0.09 [-0.20, +0.03] | +0.34 [+0.18, +0.50] | +0.27 [-0.14, +0.68] | +0.14 [-0.33, +0.60] |

## Depth effect on flips after controlling for BF16 confidence (pts, [95% CI])

Positive = deep items flip more than equally-confident shallow items.

| Model | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| qwen3-0.6b | +0.3 [-4.8, +4.4] | +3.2 [-2.4, +7.9] | +4.3 [-3.2, +10.8] | -14.0 [-23.1, -4.6] | -0.5 [-12.0, +11.2] |
| qwen3-1.7b | -5.3 [-11.4, -0.4] | -12.6 [-21.6, -4.2] | -21.0 [-31.0, -11.4] | -31.5 [-42.3, -21.4] | +9.8 [-1.0, +20.6] |
| qwen3-4b | -0.2 [-1.4, +0.8] | -3.0 [-8.0, +0.7] | -5.6 [-11.9, -0.7] | -6.5 [-15.2, +2.2] | +1.0 [-9.5, +11.7] |

## Predicted-label shares (gold is 1/3 each): quantization shifts the label prior

| Model | bf16 | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|---|
| qwen3-0.6b | T54% F2% U44% | T50% F4% U46% | T44% F1% U55% | T73% F0% U27% | T37% F57% U6% | T59% F11% U31% |
| qwen3-1.7b | T15% F6% U79% | T15% F5% U80% | T15% F23% U62% | T25% F2% U72% | T5% F0% U95% | T0% F83% U17% |
| qwen3-4b | T41% F48% U11% | T43% F44% U13% | T46% F43% U11% | T41% F46% U13% | T28% F43% U29% | T26% F59% U14% |
