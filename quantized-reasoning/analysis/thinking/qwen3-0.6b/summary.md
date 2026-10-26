# Thinking-mode precision sweep — qwen3-0.6b

180 paired items (30 per depth, balanced across labels). Greedy decoding, 4,096-token budget; TRUNCATED counts as wrong. Accuracy CIs: Wilson 95%. Penalty CIs: paired bootstrap, 10,000 resamples within depth.

## Accuracy by depth, % [95% CI]

| Depth | bf16 | 8bit | 4bit | 3bit |
|---|---|---|---|---|
| 0 | 77 [59, 88] | 87 [70, 95] | 63 [46, 78] | 0 [0, 11] |
| 1 | 80 [63, 90] | 80 [63, 90] | 63 [46, 78] | 0 [0, 11] |
| 2 | 80 [63, 90] | 80 [63, 90] | 43 [27, 61] | 0 [0, 11] |
| 3 | 40 [25, 58] | 43 [27, 61] | 17 [7, 34] | 0 [0, 11] |
| 4 | 37 [22, 54] | 37 [22, 54] | 10 [3, 26] | 0 [0, 11] |
| 5 | 40 [25, 58] | 43 [27, 61] | 7 [2, 21] | 0 [0, 11] |

## Penalty Δ = A(BF16) − A(b), points [95% CI]

| Depth | 8bit | 4bit | 3bit |
|---|---|---|---|
| 0 | -10.0 [-23.3, +0.0] | +13.3 [-3.3, +30.0] | +76.7 [+60.0, +90.0] |
| 1 | +0.0 [-16.7, +16.7] | +16.7 [-3.3, +36.7] | +80.0 [+63.3, +93.3] |
| 2 | +0.0 [-16.7, +16.7] | +36.7 [+16.7, +56.7] | +80.0 [+63.3, +93.3] |
| 3 | -3.3 [-16.7, +10.0] | +23.3 [+6.7, +40.0] | +40.0 [+23.3, +56.7] |
| 4 | +0.0 [-20.0, +20.0] | +26.7 [+6.7, +46.7] | +36.7 [+20.0, +53.3] |
| 5 | -3.3 [-20.0, +13.3] | +33.3 [+16.7, +50.0] | +40.0 [+23.3, +56.7] |

## Does the penalty grow with depth?

| Precision | Overall Δ (pts) | Slope of Δ per depth (pts) | Δ(d5) − Δ(d0) (pts) |
|---|---|---|---|
| 8bit | -2.8 [-9.4, +3.9] | +0.86 [-2.76, +4.48] | +6.7 [-13.3, +26.7] |
| 4bit | +25.0 [+17.2, +32.8] | +3.33 [-1.14, +7.71] | +20.0 [-6.7, +43.3] |
| 3bit | +58.9 [+52.2, +65.6] | -10.10 [-14.00, -6.19] | -36.7 [-60.0, -13.3] |

## Thinking length (mean generated tokens) and truncations

| Depth | bf16 | 8bit | 4bit | 3bit |
|---|---|---|---|---|
| 0 | 514 (0) | 579 (0) | 1385 (3) | 4096 (30) |
| 1 | 947 (1) | 955 (0) | 1639 (6) | 4096 (30) |
| 2 | 986 (0) | 1180 (1) | 1999 (6) | 4096 (30) |
| 3 | 1201 (0) | 1380 (1) | 2745 (14) | 4096 (30) |
| 4 | 1484 (0) | 1473 (0) | 2768 (14) | 4096 (30) |
| 5 | 1333 (0) | 1365 (0) | 3047 (19) | 4096 (30) |

## Automatic checks

- floor: bf16 depth 3 95% CI [0.25, 0.58] includes chance (0.33)
- floor: bf16 depth 4 95% CI [0.22, 0.54] includes chance (0.33)
- floor: bf16 depth 5 95% CI [0.25, 0.58] includes chance (0.33)
- floor: 8bit depth 3 95% CI [0.27, 0.61] includes chance (0.33)
- floor: 8bit depth 4 95% CI [0.22, 0.54] includes chance (0.33)
- floor: 8bit depth 5 95% CI [0.27, 0.61] includes chance (0.33)
- floor: 4bit depth 2 95% CI [0.27, 0.61] includes chance (0.33)
- floor: 4bit depth 3 95% CI [0.07, 0.34] includes chance (0.33)
- floor: 4bit depth 4 95% CI [0.03, 0.26] includes chance (0.33)
- floor: 4bit depth 5 95% CI [0.02, 0.21] includes chance (0.33)
- floor: 3bit depth 0 95% CI [0.00, 0.11] includes chance (0.33)
- floor: 3bit depth 1 95% CI [0.00, 0.11] includes chance (0.33)
- floor: 3bit depth 2 95% CI [0.00, 0.11] includes chance (0.33)
- floor: 3bit depth 3 95% CI [0.00, 0.11] includes chance (0.33)
- floor: 3bit depth 4 95% CI [0.00, 0.11] includes chance (0.33)
- floor: 3bit depth 5 95% CI [0.00, 0.11] includes chance (0.33)

## Wrong reasoning or failure to finish?

Accuracy on the items where the quantized model closed </think> with a parseable answer (strict parser), compared with BF16 on the same items.

| Precision | Items finished | Acc (quantized) | Acc (BF16, same items) | Truncated by depth 0→5 |
|---|---|---|---|---|
| 8bit | 177 | 63% | 59% | 0 / 0 / 1 / 1 / 0 / 0 |
| 4bit | 90 | 68% | 66% | 3 / 6 / 6 / 14 / 14 / 19 |
| 3bit | 0 | — | — | 30 / 30 / 30 / 30 / 30 / 30 |

## Sensitivity: lenient answer parsing (first True/False/Unknown word after </think>)

| Depth | bf16 | 8bit | 4bit | 3bit |
|---|---|---|---|---|
| 0 | 77 | 87 | 73 | 0 |
| 1 | 80 | 80 | 63 | 0 |
| 2 | 80 | 80 | 57 | 0 |
| 3 | 40 | 43 | 27 | 0 |
| 4 | 37 | 37 | 13 | 0 |
| 5 | 40 | 43 | 10 | 0 |

| Precision | Overall Δ (pts) | Slope of Δ per depth (pts) |
|---|---|---|
| 8bit | -2.8 [-9.4, +3.9] | +0.86 [-2.76, +4.48] |
| 4bit | +18.3 [+10.0, +26.7] | +4.10 [-0.10, +8.29] |
| 3bit | +58.9 [+52.2, +65.6] | -10.10 [-14.00, -6.19] |
