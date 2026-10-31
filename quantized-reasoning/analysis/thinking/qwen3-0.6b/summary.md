# Thinking-mode precision sweep — qwen3-0.6b

180 paired items (30 per depth, balanced across labels). Greedy decoding, 4,096-token budget; TRUNCATED counts as wrong. Accuracy CIs: Wilson 95%. Penalty CIs: paired bootstrap, 10,000 resamples within depth.

## Accuracy by depth, % [95% CI]

| Depth | bf16 | 8bit | 4bit | 3bit | gptq4 | gptq4b |
|---|---|---|---|---|---|---|
| 0 | 77 [59, 88] | 87 [70, 95] | 63 [46, 78] | 0 [0, 11] | 83 [66, 93] | 83 [66, 93] |
| 1 | 80 [63, 90] | 80 [63, 90] | 63 [46, 78] | 0 [0, 11] | 70 [52, 83] | 70 [52, 83] |
| 2 | 80 [63, 90] | 80 [63, 90] | 43 [27, 61] | 0 [0, 11] | 57 [39, 73] | 57 [39, 73] |
| 3 | 40 [25, 58] | 43 [27, 61] | 17 [7, 34] | 0 [0, 11] | 27 [14, 44] | 27 [14, 44] |
| 4 | 37 [22, 54] | 37 [22, 54] | 10 [3, 26] | 0 [0, 11] | 20 [10, 37] | 20 [10, 37] |
| 5 | 40 [25, 58] | 43 [27, 61] | 7 [2, 21] | 0 [0, 11] | 20 [10, 37] | 20 [10, 37] |

## Penalty Δ = A(BF16) − A(b), points [95% CI]

| Depth | 8bit | 4bit | 3bit | gptq4 | gptq4b |
|---|---|---|---|---|---|
| 0 | -10.0 [-23.3, +0.0] | +13.3 [-3.3, +30.0] | +76.7 [+60.0, +90.0] | -6.7 [-20.0, +6.7] | -6.7 [-20.0, +6.7] |
| 1 | +0.0 [-16.7, +16.7] | +16.7 [-3.3, +36.7] | +80.0 [+63.3, +93.3] | +10.0 [-10.0, +30.0] | +10.0 [-10.0, +30.0] |
| 2 | +0.0 [-16.7, +16.7] | +36.7 [+16.7, +56.7] | +80.0 [+63.3, +93.3] | +23.3 [+3.3, +43.3] | +23.3 [+3.3, +43.3] |
| 3 | -3.3 [-16.7, +10.0] | +23.3 [+6.7, +40.0] | +40.0 [+23.3, +56.7] | +13.3 [+0.0, +30.0] | +13.3 [+0.0, +30.0] |
| 4 | +0.0 [-20.0, +20.0] | +26.7 [+6.7, +46.7] | +36.7 [+20.0, +53.3] | +16.7 [+0.0, +33.3] | +16.7 [+0.0, +33.3] |
| 5 | -3.3 [-20.0, +13.3] | +33.3 [+16.7, +50.0] | +40.0 [+23.3, +56.7] | +20.0 [-3.3, +40.0] | +20.0 [-3.3, +40.0] |

## Does the penalty grow with depth?

| Precision | Overall Δ (pts) | Slope of Δ per depth (pts) | Δ(d5) − Δ(d0) (pts) |
|---|---|---|---|
| 8bit | -2.8 [-9.4, +3.9] | +0.86 [-2.76, +4.48] | +6.7 [-13.3, +26.7] |
| 4bit | +25.0 [+17.2, +32.8] | +3.33 [-1.14, +7.71] | +20.0 [-6.7, +43.3] |
| 3bit | +58.9 [+52.2, +65.6] | -10.10 [-14.00, -6.19] | -36.7 [-60.0, -13.3] |
| gptq4 | +12.8 [+5.6, +20.0] | +4.10 [-0.19, +8.29] | +26.7 [+0.0, +50.0] |
| gptq4b | +12.8 [+5.6, +20.0] | +4.10 [-0.19, +8.29] | +26.7 [+0.0, +50.0] |

## Thinking length (mean generated tokens) and truncations

| Depth | bf16 | 8bit | 4bit | 3bit | gptq4 | gptq4b |
|---|---|---|---|---|---|---|
| 0 | 514 (0) | 579 (0) | 1385 (3) | 4096 (30) | 965 (3) | 965 (3) |
| 1 | 947 (1) | 955 (0) | 1639 (6) | 4096 (30) | 1346 (5) | 1346 (5) |
| 2 | 986 (0) | 1180 (1) | 1999 (6) | 4096 (30) | 1722 (7) | 1722 (7) |
| 3 | 1201 (0) | 1380 (1) | 2745 (14) | 4096 (30) | 2197 (10) | 2197 (10) |
| 4 | 1484 (0) | 1473 (0) | 2768 (14) | 4096 (30) | 2145 (8) | 2145 (8) |
| 5 | 1333 (0) | 1365 (0) | 3047 (19) | 4096 (30) | 2006 (6) | 2006 (6) |

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
- floor: gptq4 depth 3 95% CI [0.14, 0.44] includes chance (0.33)
- floor: gptq4 depth 4 95% CI [0.10, 0.37] includes chance (0.33)
- floor: gptq4 depth 5 95% CI [0.10, 0.37] includes chance (0.33)
- floor: gptq4b depth 3 95% CI [0.14, 0.44] includes chance (0.33)
- floor: gptq4b depth 4 95% CI [0.10, 0.37] includes chance (0.33)
- floor: gptq4b depth 5 95% CI [0.10, 0.37] includes chance (0.33)

## Wrong reasoning or failure to finish?

Accuracy on the items where the quantized model closed </think> with a parseable answer (strict parser), compared with BF16 on the same items.

| Precision | Items finished | Acc (quantized) | Acc (BF16, same items) | Truncated by depth 0→5 |
|---|---|---|---|---|
| 8bit | 177 | 63% | 59% | 0 / 0 / 1 / 1 / 0 / 0 |
| 4bit | 90 | 68% | 66% | 3 / 6 / 6 / 14 / 14 / 19 |
| 3bit | 0 | — | — | 30 / 30 / 30 / 30 / 30 / 30 |
| gptq4 | 141 | 59% | 62% | 3 / 5 / 7 / 10 / 8 / 6 |
| gptq4b | 141 | 59% | 62% | 3 / 5 / 7 / 10 / 8 / 6 |

## Sensitivity: lenient answer parsing (first True/False/Unknown word after </think>)

| Depth | bf16 | 8bit | 4bit | 3bit | gptq4 | gptq4b |
|---|---|---|---|---|---|---|
| 0 | 77 | 87 | 73 | 0 | 83 | 83 |
| 1 | 80 | 80 | 63 | 0 | 70 | 70 |
| 2 | 80 | 80 | 57 | 0 | 57 | 57 |
| 3 | 40 | 43 | 27 | 0 | 27 | 27 |
| 4 | 37 | 37 | 13 | 0 | 20 | 20 |
| 5 | 40 | 43 | 10 | 0 | 20 | 20 |

| Precision | Overall Δ (pts) | Slope of Δ per depth (pts) |
|---|---|---|
| 8bit | -2.8 [-9.4, +3.9] | +0.86 [-2.76, +4.48] |
| 4bit | +18.3 [+10.0, +26.7] | +4.10 [-0.10, +8.29] |
| 3bit | +58.9 [+52.2, +65.6] | -10.10 [-14.00, -6.19] |
| gptq4 | +12.8 [+5.6, +20.0] | +4.10 [-0.19, +8.29] |
| gptq4b | +12.8 [+5.6, +20.0] | +4.10 [-0.19, +8.29] |
