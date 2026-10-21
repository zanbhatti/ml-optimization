# Forced-choice precision sweep — Qwen3-4B-Instruct-2507

540 paired ProofWriter items; label chosen by argmax over the log-probs of the True/False/Unknown tokens at the first answer position (no format failures possible). Margin = log p(gold) − max log p(wrong), in nats. CIs: Wilson (accuracy) and paired bootstrap with 10,000 resamples within depth (everything else).

## Accuracy by depth, %

| Depth | bf16 | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|---|
| 0 | 90.0 | 90.0 | 88.9 | 85.6 | 80.0 | 73.3 |
| 1 | 53.3 | 52.2 | 54.4 | 51.1 | 43.3 | 54.4 |
| 2 | 43.3 | 43.3 | 44.4 | 48.9 | 42.2 | 47.8 |
| 3 | 45.6 | 46.7 | 44.4 | 42.2 | 40.0 | 51.1 |
| 4 | 44.4 | 46.7 | 45.6 | 41.1 | 42.2 | 45.6 |
| 5 | 44.4 | 46.7 | 45.6 | 42.2 | 34.4 | 45.6 |

## Dose-response and depth interaction (all items)

Slope = change in the penalty per +1 depth. Positive with a CI above 0 means the penalty grows with depth.

| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |
|---|---|---|---|---|---|
| 8bit | -0.7 [-1.7, +0.2] | -0.10 [-0.14, -0.05] | -0.63 [-1.24, -0.16] | -0.030 [-0.056, -0.004] | +0.0006 [-0.0002, +0.0016] |
| 6bit | -0.4 [-1.9, +1.1] | +0.07 [-0.02, +0.17] | -0.25 [-1.21, +0.67] | +0.047 [-0.011, +0.105] | +0.0024 [-0.0022, +0.0077] |
| 5bit | +1.7 [-0.9, +4.3] | +0.59 [+0.37, +0.81] | +0.03 [-1.40, +1.49] | -0.132 [-0.268, +0.004] | +0.0109 [-0.0148, +0.0366] |
| 4bit | +6.5 [+3.0, +10.0] | +1.17 [+0.79, +1.55] | -0.54 [-2.44, +1.43] | -0.262 [-0.479, -0.041] | +0.0700 [+0.0003, +0.1485] |
| 3bit | +0.6 [-4.6, +5.7] | +1.94 [+1.17, +2.69] | -2.57 [-5.56, +0.38] | -0.655 [-1.117, -0.201] | +0.2439 [+0.0406, +0.4497] |

## Robustness: True/False items only (depth is a proof length only for provable items)

| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |
|---|---|---|---|---|---|
| 8bit | -1.1 [-2.2, -0.3] | -0.24 [-0.29, -0.19] | -0.76 [-1.62, -0.14] | -0.022 [-0.051, +0.008] | +0.0009 [-0.0003, +0.0022] |
| 6bit | +0.0 [-1.7, +1.7] | +0.28 [+0.17, +0.38] | -0.95 [-2.00, -0.05] | +0.041 [-0.023, +0.104] | +0.0007 [-0.0032, +0.0046] |
| 5bit | +7.5 [+4.7, +10.6] | +1.88 [+1.68, +2.08] | +0.05 [-1.81, +1.90] | -0.096 [-0.225, +0.029] | +0.0031 [-0.0320, +0.0365] |
| 4bit | +2.8 [-1.1, +6.7] | +0.22 [-0.17, +0.61] | -2.48 [-4.71, -0.29] | -0.439 [-0.662, -0.217] | +0.0379 [-0.0228, +0.1020] |
| 3bit | -25.6 [-30.8, -20.6] | -2.04 [-2.75, -1.34] | -3.62 [-6.48, -0.71] | -0.656 [-1.078, -0.228] | +0.3515 [+0.1240, +0.5791] |

## Margin loss ΔM by depth (nats, [95% CI])

| Depth | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| 0 | +0.00 [-0.12, +0.12] | -0.26 [-0.53, +0.01] | +1.20 [+0.62, +1.82] | +2.65 [+1.81, +3.51] | +5.92 [+4.04, +7.76] |
| 1 | -0.05 [-0.16, +0.07] | +0.17 [-0.06, +0.40] | +0.53 [-0.03, +1.11] | +1.31 [+0.48, +2.14] | +1.12 [-0.89, +3.11] |
| 2 | -0.16 [-0.27, -0.06] | +0.18 [-0.02, +0.38] | +0.10 [-0.38, +0.57] | +0.39 [-0.58, +1.40] | +1.61 [-0.47, +3.69] |
| 3 | -0.06 [-0.18, +0.06] | +0.15 [-0.08, +0.37] | +1.08 [+0.59, +1.58] | +0.98 [+0.10, +1.86] | +0.36 [-1.32, +2.02] |
| 4 | -0.16 [-0.26, -0.05] | +0.08 [-0.15, +0.31] | +0.48 [-0.10, +1.06] | +0.48 [-0.52, +1.38] | +0.97 [-0.56, +2.54] |
| 5 | -0.16 [-0.26, -0.07] | +0.12 [-0.11, +0.35] | +0.11 [-0.45, +0.67] | +1.20 [+0.21, +2.22] | +1.68 [-0.24, +3.69] |

## Flip rate vs BF16 by depth (% of items whose answer changes)

| Depth | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| 0 | 0.0 | 1.1 | 8.9 | 12.2 | 25.6 |
| 1 | 2.2 | 5.6 | 10.0 | 24.4 | 52.2 |
| 2 | 0.0 | 3.3 | 8.9 | 21.1 | 56.7 |
| 3 | 3.3 | 3.3 | 14.4 | 23.3 | 46.7 |
| 4 | 4.4 | 4.4 | 17.8 | 26.7 | 54.4 |
| 5 | 2.2 | 6.7 | 8.9 | 22.2 | 56.7 |

## Depth vs BF16 confidence as the driver of answer flips

BF16 median top-2 label gap (nats) by depth: d0 12.4, d1 6.2, d2 4.2, d3 4.1, d4 3.5, d5 4.9.
Deep effect = extra flip probability for depth ≥ 1 after controlling for that gap (linear probability model, 2,000 item bootstraps). Prediction shares show label-prior shifts (BF16: True 39%, False 19%, Unknown 42%).

| Precision | Flip rate | Deep effect (pts) | Pred. True | Pred. False | Pred. Unknown |
|---|---|---|---|---|---|
| 8bit | 2.0% | -0.4 [-1.3, +0.2] | 38% | 20% | 41% |
| 6bit | 4.1% | -2.3 [-6.1, +0.1] | 40% | 19% | 41% |
| 5bit | 11.5% | -12.0 [-19.4, -5.4] | 36% | 14% | 51% |
| 4bit | 21.7% | -10.2 [-18.5, -2.1] | 50% | 16% | 34% |
| 3bit | 48.7% | +13.7 [+1.8, +25.6] | 43% | 54% | 4% |
