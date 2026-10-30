# Greedy-decoding check — Qwen3-0.6B thinking mode

Δ = BF16 accuracy − quantized accuracy (points; positive = quantization hurts). 95% CIs from a paired bootstrap within depth (10,000 resamples). Slope = change in Δ per extra reasoning step.

## Penalty under each decoding setup (strict parser)

| Setup | Precision | Items | Δ overall | Slope per depth | Δ by depth 0–5 |
|---|---|---|---|---|---|
| greedy, 4k tokens | 4bit | 180 | +25.0 [+17.2, +32.8] | +3.33 [-1.14, +7.71] | +13 / +17 / +37 / +23 / +27 / +33 |
| greedy, 4k tokens | gptq4 | 180 | +12.8 [+5.6, +20.0] | +4.10 [-0.19, +8.29] | -7 / +10 / +23 / +13 / +17 / +20 |
| greedy, 8k tokens | 4bit | 180 | +25.6 [+17.8, +33.3] | +3.05 [-1.34, +7.43] | +13 / +20 / +37 / +23 / +27 / +33 |
| greedy, 8k tokens | gptq4 | 180 | +13.3 [+6.1, +20.6] | +3.81 [-0.38, +8.00] | -7 / +13 / +23 / +13 / +17 / +20 |
| sampled (2 seeds), 4k tokens | 4bit | 180 | +35.6 [+29.2, +41.9] | -0.57 [-4.29, +3.05] | +33 / +28 / +52 / +37 / +35 / +28 |

## Penalty under each decoding setup (lenient parser)

| Setup | Precision | Items | Δ overall | Slope per depth | Δ by depth 0–5 |
|---|---|---|---|---|---|
| greedy, 4k tokens | 4bit | 180 | +18.3 [+10.0, +26.7] | +4.10 [-0.10, +8.29] | +3 / +17 / +23 / +13 / +23 / +30 |
| greedy, 4k tokens | gptq4 | 180 | +12.8 [+5.6, +20.0] | +4.10 [-0.19, +8.29] | -7 / +10 / +23 / +13 / +17 / +20 |
| greedy, 8k tokens | 4bit | 180 | +17.8 [+9.4, +26.1] | +3.05 [-1.24, +7.24] | +3 / +20 / +23 / +13 / +20 / +27 |
| greedy, 8k tokens | gptq4 | 180 | +13.3 [+6.1, +20.6] | +3.81 [-0.38, +8.00] | -7 / +13 / +23 / +13 / +17 / +20 |
| sampled (2 seeds), 4k tokens | 4bit | 180 | +17.8 [+11.1, +24.4] | -1.14 [-4.86, +2.57] | +15 / +18 / +27 / +22 / +15 / +10 |

## Accuracy and truncation by depth (strict parser)

| Setup | Precision | Accuracy % by depth 0–5 | Truncated % by depth 0–5 | Truncated overall |
|---|---|---|---|---|
| greedy, 4k tokens | bf16 | 77 / 80 / 80 / 40 / 37 / 40 | 0 / 3 / 0 / 0 / 0 / 0 | 1% |
| greedy, 4k tokens | 4bit | 63 / 63 / 43 / 17 / 10 / 7 | 10 / 20 / 20 / 47 / 47 / 63 | 34% |
| greedy, 4k tokens | gptq4 | 83 / 70 / 57 / 27 / 20 / 20 | 10 / 17 / 23 / 33 / 27 / 20 | 22% |
| greedy, 8k tokens | bf16 | 77 / 83 / 80 / 40 / 37 / 40 | 0 / 0 / 0 / 0 / 0 / 0 | 0% |
| greedy, 8k tokens | 4bit | 63 / 63 / 43 / 17 / 10 / 7 | 10 / 17 / 20 / 43 / 40 / 50 | 30% |
| greedy, 8k tokens | gptq4 | 83 / 70 / 57 / 27 / 20 / 20 | 10 / 17 / 23 / 33 / 27 / 20 | 22% |
| sampled (2 seeds), 4k tokens | bf16 | 88 / 83 / 87 / 48 / 47 / 38 | 0 / 0 / 0 / 0 / 0 / 2 | 0% |
| sampled (2 seeds), 4k tokens | 4bit | 55 / 55 / 35 / 12 / 12 / 10 | 8 / 10 / 15 / 23 / 15 / 23 | 16% |

## What happened to each item (lenient parser)

| Run | Precision | Correct | Wrong label | Truncated | No label | Accuracy on answered |
|---|---|---|---|---|---|---|
| greedy, 4k | bf16 | 106 | 73 | 1 | 0 | 59% |
| greedy, 4k | 4bit | 73 | 45 | 62 | 0 | 62% |
| greedy, 4k | gptq4 | 83 | 58 | 39 | 0 | 59% |
| sampled seed 1 | bf16 | 122 | 57 | 1 | 0 | 68% |
| sampled seed 1 | 4bit | 83 | 66 | 29 | 2 | 56% |
| sampled seed 2 | bf16 | 113 | 67 | 0 | 0 | 63% |
| sampled seed 2 | 4bit | 88 | 64 | 28 | 0 | 58% |
