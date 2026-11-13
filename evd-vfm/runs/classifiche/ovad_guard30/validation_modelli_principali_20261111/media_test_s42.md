# OVAD mean of the two tests

Seed 42. Metrics in percent; values in CSV/JSON use fractions.

| Metric | LoRA → VPT-shallow n=8 | LoRA r=32 | VPT-shallow n=8 | Attentive Probe |
|---|---:|---:|---:|---:|
| protocol mean | 86.1879 | 85.4358 | 84.4741 | 84.1717 |
| row · AUC | 93.7636 | 93.6654 | 93.0955 | 93.0696 |
| row · AP | 88.2734 | 87.6954 | 86.9711 | 86.8899 |
| row · AUC_a macro | 91.8914 | 91.9435 | 88.9673 | 89.3969 |
| row · AP_a macro | 93.0601 | 93.0373 | 91.7346 | 92.1168 |
| row · AUC_a micro | 89.3128 | 89.0189 | 86.8290 | 87.0723 |
| row · AP_a micro | 92.4970 | 92.2544 | 90.7203 | 90.7421 |
| row · precision | 76.8674 | 74.3977 | 77.5125 | 76.1288 |
| row · recall | 81.2599 | 81.4939 | 78.8854 | 78.3705 |
| row · F1 | 78.4787 | 76.8227 | 77.2854 | 76.5039 |
| row · FPR | 9.9924 | 11.5841 | 9.3921 | 10.1594 |
| video · precision | 78.5886 | 77.0033 | 77.8328 | 75.5639 |
| video · recall | 90.1587 | 88.6772 | 84.2460 | 83.9021 |
| video · F1 | 83.8828 | 82.2558 | 80.5222 | 79.2348 |
| video · FPR | 15.8102 | 17.1038 | 15.6028 | 17.5811 |
| event · recall | 79.1031 | 79.5115 | 76.3664 | 76.9847 |

Each value is the arithmetic mean of the UCF-Crime and MIVIA-Crime test metrics. This is not a pooled evaluation or an average over seeds. Counts and thresholds are not averaged.
