# Mean of the UCF/MIVIA means — seed 42

Mean of means = (UCF mean + MIVIA mean) / 2. It is not the mean of the AUCs.

| # | Model | UCF mean | MIVIA mean | Mean of means |
|---|---|---|---|---|
| 1 | VPT-shallow n=8 on fused LoRA r=32 | 85.89 | 86.49 | 86.19 |
| 2 | LoRA r=32 on frozen VPT-shallow n=8 | 84.64 | 86.62 | 85.63 |
| 3 | VPT-deep n=8 on fused LoRA r=32 | 85.24 | 85.91 | 85.58 |
| 4 | VPT-deep n=8 on fused LoRA r=32 (prior val-best) | 85.13 | 85.96 | 85.54 |
| 5 | LoRA r=32 | 85.45 | 85.43 | 85.44 |
| 6 | VPT-shallow n=8 on fused DoRA r=32 (prior val-best) | 85.94 | 84.69 | 85.32 |
| 7 | DoRA r=32 | 85.46 | 84.96 | 85.21 |
| 8 | DoRA r=32 + VPT-deep n=8 | 85.83 | 84.39 | 85.11 |
| 9 | VPT-deep n=8 on fused DoRA r=16 (1+2 ep.) | 84.21 | 85.98 | 85.10 |
| 10 | DoRA r=32 + VPT-shallow n=8 | 83.70 | 86.34 | 85.02 |
| 11 | LoRA r=64 | 83.91 | 86.11 | 85.01 |
| 12 | LoRA r=8 | 84.97 | 84.94 | 84.95 |
| 13 | LoRA r=32 + VPT-shallow n=8 | 85.08 | 84.71 | 84.90 |
| 14 | VPT-shallow n=8 on fused DoRA r=32 | 85.77 | 84.00 | 84.89 |
| 15 | VPT-deep n=8 on fused DoRA r=32 (prior val-best) | 85.94 | 83.76 | 84.85 |
| 16 | DoRA r=32 on frozen VPT-deep n=8 | 84.79 | 84.74 | 84.76 |
| 17 | DoRA r=32 on frozen VPT-shallow n=8 | 84.34 | 84.98 | 84.66 |
| 18 | LoRA r=16 | 84.84 | 84.37 | 84.60 |
| 19 | DoRA r=8 | 85.00 | 84.03 | 84.52 |
| 20 | VPT-shallow n=8 | 85.71 | 83.24 | 84.47 |
| 21 | LoRA r=32 on frozen VPT-shallow n=8 (prior val-best) | 84.07 | 84.73 | 84.40 |
| 22 | VPT-deep n=8 | 86.04 | 82.67 | 84.35 |
| 23 | VPT-deep n=8 on fused DoRA r=32 | 85.05 | 83.54 | 84.30 |
| 24 | DoRA r=64 | 84.53 | 84.02 | 84.27 |
| 25 | DoRA r=16 on frozen VPT-deep n=8 (1+2 ep.) | 85.54 | 82.94 | 84.24 |
| 26 | VPT-shallow n=8 on fused LoRA r=32 (prior val-best) | 83.97 | 84.50 | 84.24 |
| 27 | LoRA r=32 on frozen VPT-deep n=8 | 84.61 | 83.82 | 84.21 |
| 28 | DoRA r=16 on frozen VPT-shallow n=8 (1+2 ep.) | 84.31 | 84.09 | 84.20 |
| 29 | Attentive probe | 85.52 | 82.82 | 84.17 |
| 30 | DoRA r=32 on frozen VPT-deep n=8 (prior val-best) | 84.48 | 83.72 | 84.10 |
| 31 | VPT-shallow n=8 on fused DoRA r=16 (1+2 ep.) | 84.90 | 83.24 | 84.07 |
| 32 | LoRA r=32 + VPT-deep n=8 | 83.46 | 84.57 | 84.02 |
| 33 | VPT-deep n=64 | 84.46 | 83.41 | 83.94 |
| 34 | VPT-shallow n=576 | 85.16 | 82.59 | 83.88 |
| 35 | DoRA r=16 | 85.75 | 81.80 | 83.77 |
| 36 | VPT-shallow n=192 | 84.76 | 82.73 | 83.75 |
| 37 | VPT-shallow n=64 | 85.46 | 81.99 | 83.72 |
| 38 | LoRA r=32 on frozen VPT-deep n=8 (prior val-best) | 83.81 | 83.57 | 83.69 |
| 39 | DoRA r=32 on frozen VPT-shallow n=8 (prior val-best) | 84.43 | 82.93 | 83.68 |
| 40 | VPT-deep n=192 | 83.63 | 81.33 | 82.48 |
| 41 | VPT-deep n=576 | 80.83 | 77.31 | 79.07 |
| 42 | Linear probe | 72.67 | 75.13 | 73.90 |
