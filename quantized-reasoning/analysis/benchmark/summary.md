# Speed and memory by size and precision (Apple M3 Pro, 18 GB, MLX)

Decode = generating tokens one at a time; prefill = reading the prompt. Medians of 3 runs, 128 decoded tokens each. Ceiling = 150 GB/s ÷ weight size: the fastest decode could go if it did nothing but read every weight once per token.

| Model | Precision | Weights (GB) | Bits/weight | Peak mem @4k (GB) | Prefill tok/s @512 | Prefill tok/s @4k | Decode tok/s @512 | Decode tok/s @4k | Decode speedup vs BF16 | Ceiling tok/s | % of ceiling |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | bf16 | 1.19 | 16.0 | 2.39 | 3002 | 2589 | 80 | 62 | 1.00× | 126 | 63% |
| Qwen3-0.6B | 8bit | 0.63 | 8.5 | 1.96 | 3125 | 2452 | 127 | 89 | 1.60× | 237 | 54% |
| Qwen3-0.6B | 4bit | 0.34 | 4.5 | 1.71 | 3117 | 2465 | 184 | 112 | 2.31× | 447 | 41% |
| Qwen3-0.6B | 3bit | 0.26 | 3.5 | 1.64 | 3102 | 2453 | 208 | 120 | 2.61× | 575 | 36% |
| Qwen3-0.6B | gptq4 | 0.34 | 4.5 | 1.71 | 3120 | 2461 | 183 | 110 | 2.30× | 447 | 41% |
| Qwen3-1.7B | bf16 | 3.44 | 16.0 | 4.46 | 1127 | 1135 | 31 | 28 | 1.00× | 44 | 70% |
| Qwen3-1.7B | 8bit | 1.83 | 8.5 | 3.10 | 1328 | 1243 | 66 | 55 | 2.17× | 82 | 81% |
| Qwen3-1.7B | 4bit | 0.97 | 4.5 | 2.23 | 1358 | 1231 | 112 | 75 | 3.68× | 155 | 73% |
| Qwen3-1.7B | 3bit | 0.75 | 3.5 | 2.01 | 1344 | 1113 | 134 | 87 | 4.40× | 199 | 67% |
| Qwen3-4B | bf16 | 8.04 | 16.0 | 9.26 | 525 | 517 | 16 | 15 | 1.00× | 19 | 87% |
| Qwen3-4B | 8bit | 4.27 | 8.5 | 5.66 | 519 | 481 | 30 | 27 | 1.83× | 35 | 85% |
| Qwen3-4B | 4bit | 2.26 | 4.5 | 3.67 | 530 | 485 | 52 | 43 | 3.20× | 66 | 79% |
| Qwen3-4B | 3bit | 1.76 | 3.5 | 3.16 | 535 | 485 | 64 | 51 | 3.91× | 85 | 75% |
