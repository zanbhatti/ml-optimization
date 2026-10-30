# Forced-choice precision sweep — Qwen3-1.7B

540 paired ProofWriter items; label chosen by argmax over the log-probs of the True/False/Unknown tokens at the first answer position (no format failures possible). Margin = log p(gold) − max log p(wrong), in nats. CIs: Wilson (accuracy) and paired bootstrap with 10,000 resamples within depth (everything else).

## Accuracy by depth, %

| Depth | bf16 | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|---|
| 0 | 83.3 | 81.1 | 86.7 | 67.8 | 58.9 | 37.8 |
| 1 | 35.6 | 37.8 | 33.3 | 37.8 | 35.6 | 35.6 |
| 2 | 36.7 | 36.7 | 38.9 | 36.7 | 34.4 | 31.1 |
| 3 | 38.9 | 35.6 | 40.0 | 40.0 | 34.4 | 28.9 |
| 4 | 35.6 | 38.9 | 41.1 | 37.8 | 34.4 | 26.7 |
| 5 | 37.8 | 37.8 | 41.1 | 40.0 | 33.3 | 33.3 |

## Dose-response and depth interaction (all items)

Slope = change in the penalty per +1 depth. Positive with a CI above 0 means the penalty grows with depth.

| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |
|---|---|---|---|---|---|
| 8bit | +0.0 [-1.5, +1.5] | -0.02 [-0.11, +0.07] | -0.32 [-1.24, +0.57] | -0.174 [-0.224, -0.121] | -0.0006 [-0.0031, +0.0017] |
| 6bit | -2.2 [-5.7, +1.3] | -0.43 [-0.61, -0.24] | -0.63 [-2.60, +1.40] | -0.101 [-0.214, +0.011] | -0.0049 [-0.0207, +0.0098] |
| 5bit | +1.3 [-1.9, +4.4] | +0.08 [-0.18, +0.34] | -2.57 [-4.51, -0.67] | -0.220 [-0.369, -0.072] | +0.0184 [-0.0161, +0.0552] |
| 4bit | +6.1 [+3.0, +9.3] | +4.72 [+3.58, +5.84] | -2.70 [-4.60, -0.73] | +0.252 [-0.387, +0.906] | -0.1414 [-0.3464, +0.0657] |
| 3bit | +12.4 [+5.9, +18.7] | +2.12 [+1.45, +2.80] | -4.98 [-8.67, -1.27] | -1.116 [-1.532, -0.698] | -0.0256 [-0.1432, +0.0886] |

## Robustness: True/False items only (depth is a proof length only for provable items)

| Precision | Δacc overall (pts) | Δmargin overall (nats) | slope Δacc (pts/depth) | slope Δmargin (nats/depth) | slope KL (nats/depth) |
|---|---|---|---|---|---|
| 8bit | +1.1 [-0.6, +2.8] | -0.31 [-0.41, -0.22] | -0.19 [-1.29, +0.81] | -0.221 [-0.280, -0.164] | -0.0013 [-0.0048, +0.0017] |
| 6bit | -13.1 [-16.9, -9.2] | -1.68 [-1.82, -1.54] | -0.81 [-3.05, +1.43] | -0.085 [-0.176, +0.005] | -0.0054 [-0.0259, +0.0138] |
| 5bit | -1.7 [-5.3, +1.9] | -0.22 [-0.54, +0.10] | -3.90 [-6.14, -1.62] | -0.295 [-0.466, -0.119] | +0.0118 [-0.0220, +0.0473] |
| 4bit | +15.3 [+11.9, +18.9] | +13.54 [+13.09, +13.98] | -3.95 [-6.33, -1.62] | +0.522 [+0.222, +0.819] | -0.2394 [-0.5069, +0.0347] |
| 3bit | -20.3 [-26.9, -13.6] | -0.43 [-1.19, +0.38] | -7.95 [-12.00, -3.81] | -1.519 [-2.049, -0.988] | -0.1046 [-0.2556, +0.0426] |

## Margin loss ΔM by depth (nats, [95% CI])

| Depth | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| 0 | +0.96 [+0.78, +1.15] | +0.21 [-0.31, +0.74] | +1.24 [+0.71, +1.77] | +4.06 [+1.60, +6.58] | +8.96 [+6.81, +11.18] |
| 1 | -0.31 [-0.51, -0.10] | -0.74 [-1.21, -0.28] | -0.11 [-0.77, +0.55] | +4.59 [+1.89, +7.25] | +0.23 [-1.33, +1.82] |
| 2 | -0.06 [-0.25, +0.12] | -0.47 [-0.88, -0.03] | +0.04 [-0.62, +0.66] | +4.73 [+1.91, +7.44] | +1.03 [-0.49, +2.60] |
| 3 | -0.17 [-0.39, +0.06] | -0.29 [-0.72, +0.15] | -0.28 [-0.90, +0.31] | +4.33 [+1.48, +7.01] | +0.84 [-0.57, +2.27] |
| 4 | -0.30 [-0.52, -0.08] | -0.78 [-1.20, -0.36] | -0.30 [-0.99, +0.39] | +4.93 [+2.13, +7.69] | +0.91 [-0.55, +2.45] |
| 5 | -0.23 [-0.49, +0.02] | -0.51 [-0.95, -0.06] | -0.12 [-0.79, +0.57] | +5.70 [+2.87, +8.42] | +0.77 [-0.51, +2.07] |

## Flip rate vs BF16 by depth (% of items whose answer changes)

| Depth | 8bit | 6bit | 5bit | 4bit | 3bit |
|---|---|---|---|---|---|
| 0 | 4.4 | 16.7 | 23.3 | 33.3 | 74.4 |
| 1 | 2.2 | 17.8 | 15.6 | 10.0 | 80.0 |
| 2 | 0.0 | 21.1 | 13.3 | 15.6 | 85.6 |
| 3 | 4.4 | 21.1 | 15.6 | 8.9 | 80.0 |
| 4 | 5.6 | 25.6 | 20.0 | 14.4 | 85.6 |
| 5 | 2.2 | 16.7 | 15.6 | 12.2 | 77.8 |

## Depth vs BF16 confidence as the driver of answer flips

BF16 median top-2 label gap (nats) by depth: d0 6.1, d1 4.0, d2 3.8, d3 3.4, d4 3.7, d5 3.7.
Deep effect = extra flip probability for depth ≥ 1 after controlling for that gap (linear probability model, 2,000 item bootstraps). Prediction shares show label-prior shifts (BF16: True 15%, False 6%, Unknown 79%).

| Precision | Flip rate | Deep effect (pts) | Pred. True | Pred. False | Pred. Unknown |
|---|---|---|---|---|---|
| 8bit | 3.1% | -5.3 [-11.4, -0.4] | 15% | 5% | 80% |
| 6bit | 19.8% | -12.6 [-21.6, -4.2] | 15% | 23% | 62% |
| 5bit | 17.2% | -21.0 [-31.0, -11.4] | 25% | 2% | 72% |
| 4bit | 15.7% | -31.5 [-42.3, -21.4] | 5% | 0% | 95% |
| 3bit | 80.6% | +9.8 [-1.0, +20.6] | 0% | 83% | 17% |
