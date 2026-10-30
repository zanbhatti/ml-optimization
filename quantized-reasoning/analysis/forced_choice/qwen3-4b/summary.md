# Forced-choice precision sweep — Qwen3-4B

540 paired ProofWriter items; label chosen by argmax over the log-probs of the True/False/Unknown tokens at the first answer position (no format failures possible). Margin = log p(gold) − max log p(wrong), in nats. CIs: Wilson (accuracy) and paired bootstrap with 10,000 resamples within depth (everything else).

## Accuracy by depth, %

| Depth | bf16 | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|---|
| 0 | 82.2 | 82.2 | 83.3 | 82.2 | 90.0 | 73.3 |
| 1 | 52.2 | 52.2 | 51.1 | 53.3 | 53.3 | 53.3 |
| 2 | 53.3 | 55.6 | 53.3 | 54.4 | 48.9 | 51.1 |
| 3 | 42.2 | 46.7 | 46.7 | 45.6 | 50.0 | 56.7 |
| 4 | 46.7 | 46.7 | 43.3 | 44.4 | 40.0 | 45.6 |
| 5 | 46.7 | 46.7 | 47.8 | 47.8 | 38.9 | 46.7 |

## Dose-response and depth interaction (all items)

Slope = change in the penalty per +1 depth. Positive with a CI above 0 means the penalty grows with depth.

| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |
|---|---|---|---|---|---|
| 8bit | -1.1 [-2.4, +0.0] | -0.09 [-0.14, -0.03] | -0.06 [-0.67, +0.54] | -0.026 [-0.058, +0.007] | +0.0023 [+0.0003, +0.0044] |
| 6bit | -0.4 [-2.4, +1.7] | -0.09 [-0.20, +0.03] | +0.06 [-0.83, +0.95] | +0.028 [-0.035, +0.091] | +0.0106 [+0.0051, +0.0171] |
| 5bit | -0.7 [-2.8, +1.3] | +0.34 [+0.18, +0.50] | +0.06 [-1.08, +1.21] | +0.050 [-0.055, +0.149] | -0.0069 [-0.0369, +0.0153] |
| 4bit | +0.4 [-3.5, +4.3] | +0.27 [-0.14, +0.68] | +2.54 [+0.54, +4.63] | -0.291 [-0.531, -0.042] | +0.0727 [+0.0287, +0.1173] |
| 3bit | -0.6 [-4.4, +3.3] | +0.14 [-0.33, +0.60] | -1.56 [-3.62, +0.54] | -0.737 [-1.017, -0.452] | +0.0640 [-0.0177, +0.1453] |

## Robustness: True/False items only (depth is a proof length only for provable items)

| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |
|---|---|---|---|---|---|
| 8bit | -0.6 [-2.2, +0.8] | +0.10 [+0.04, +0.16] | -0.29 [-1.14, +0.52] | -0.057 [-0.090, -0.023] | +0.0041 [+0.0019, +0.0069] |
| 6bit | -0.3 [-3.1, +2.5] | -0.48 [-0.62, -0.34] | -0.05 [-1.29, +1.19] | +0.042 [-0.029, +0.113] | +0.0112 [+0.0047, +0.0194] |
| 5bit | +0.0 [-2.8, +2.8] | +0.17 [-0.00, +0.36] | -0.29 [-1.76, +1.24] | -0.026 [-0.147, +0.082] | -0.0128 [-0.0578, +0.0186] |
| 4bit | +11.9 [+7.5, +16.4] | +2.54 [+2.17, +2.91] | +2.14 [-0.19, +4.48] | -0.208 [-0.411, -0.010] | +0.1078 [+0.0567, +0.1586] |
| 3bit | -0.6 [-5.6, +4.4] | +1.05 [+0.52, +1.56] | -1.33 [-4.05, +1.33] | -0.807 [-1.118, -0.500] | +0.0593 [-0.0466, +0.1616] |

## Margin loss ΔM by depth (nats, [95% CI])

| Depth | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| 0 | +0.04 [-0.11, +0.19] | -0.29 [-0.52, -0.07] | +0.25 [-0.27, +0.86] | +1.78 [+0.85, +2.68] | +4.21 [+3.04, +5.42] |
| 1 | -0.08 [-0.20, +0.04] | -0.05 [-0.32, +0.22] | +0.38 [+0.08, +0.70] | -0.13 [-1.06, +0.79] | -0.22 [-1.55, +1.08] |
| 2 | -0.11 [-0.26, +0.03] | +0.01 [-0.33, +0.34] | -0.01 [-0.40, +0.37] | +0.58 [-0.37, +1.48] | -0.16 [-1.17, +0.83] |
| 3 | -0.17 [-0.31, -0.03] | -0.14 [-0.44, +0.18] | +0.45 [+0.11, +0.78] | -0.52 [-1.63, +0.59] | -1.65 [-2.75, -0.54] |
| 4 | -0.05 [-0.16, +0.07] | +0.09 [-0.19, +0.37] | +0.55 [+0.23, +0.88] | +0.02 [-0.86, +0.93] | -1.44 [-2.49, -0.37] |
| 5 | -0.15 [-0.29, -0.02] | -0.15 [-0.44, +0.14] | +0.41 [+0.08, +0.73] | -0.13 [-1.32, +1.02] | +0.08 [-1.06, +1.23] |

## Flip rate vs BF16 by depth (% of items whose answer changes)

| Depth | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| 0 | 0.0 | 2.2 | 4.4 | 15.6 | 20.0 |
| 1 | 5.6 | 4.4 | 10.0 | 26.7 | 34.4 |
| 2 | 3.3 | 11.1 | 7.8 | 33.3 | 30.0 |
| 3 | 6.7 | 14.4 | 12.2 | 34.4 | 36.7 |
| 4 | 2.2 | 10.0 | 8.9 | 31.1 | 40.0 |
| 5 | 3.3 | 5.6 | 8.9 | 26.7 | 30.0 |

## Depth vs BF16 confidence as the driver of answer flips

BF16 median top-2 label gap (nats) by depth: d0 8.8, d1 4.2, d2 3.4, d3 3.9, d4 3.4, d5 3.9.
Deep effect = extra flip probability for depth ≥ 1 after controlling for that gap (linear probability model, 2,000 item bootstraps). Prediction shares show label-prior shifts (BF16: True 41%, False 48%, Unknown 11%).

| Precision | Flip rate | Deep effect (pts) | Pred. True | Pred. False | Pred. Unknown |
|---|---|---|---|---|---|
| 8bit | 3.5% | -0.2 [-1.4, +0.8] | 43% | 44% | 13% |
| 6bit | 8.0% | -3.0 [-8.0, +0.7] | 46% | 43% | 11% |
| 5bit | 8.7% | -5.6 [-11.9, -0.7] | 41% | 46% | 13% |
| 4bit | 28.0% | -6.5 [-15.2, +2.2] | 28% | 43% | 29% |
| 3bit | 31.9% | +1.0 [-9.5, +11.7] | 26% | 59% | 14% |
