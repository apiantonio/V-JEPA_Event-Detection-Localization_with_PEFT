# LoRA to VPT with inherited head: Mean of the UCF and MIVIA tests

Seed 42; three VPT training epochs; OVAD guard 30 s. Metrics in percent.

| Metric | Value |
|---|---:|
| protocol mean | 86.227155 |
| row · AUC | 94.181582 |
| row · AP | 88.625056 |
| row · AUC_a macro | 91.179807 |
| row · AP_a macro | 92.439998 |
| row · AUC_a micro | 90.205941 |
| row · AP_a micro | 92.786518 |
| row · precision | 77.899160 |
| row · recall | 81.410940 |
| row · F1 | 79.104469 |
| row · FPR | 9.414046 |
| video · precision | 78.422791 |
| video · recall | 88.677249 |
| video · F1 | 83.076387 |
| video · FPR | 15.741051 |
| event · recall | 79.234733 |

Arithmetic mean of the two dataset metrics; no pooling or averaging over seeds.
