# OVAD UCF-Crime test

Seed 42. Metrics in percent; values in CSV/JSON use fractions.

| Metric | LoRA → VPT-shallow n=8 | LoRA r=32 | VPT-shallow n=8 | Attentive Probe |
|---|---:|---:|---:|---:|
| protocol mean | 85.8860 | 85.4459 | 85.7101 | 85.5243 |
| row · AUC | 94.6728 | 94.5501 | 94.4575 | 94.3023 |
| row · AP | 86.6186 | 85.9963 | 85.8212 | 85.8078 |
| row · AUC_a macro | 91.3870 | 90.7851 | 89.7112 | 90.7903 |
| row · AP_a macro | 91.7775 | 91.4394 | 90.5973 | 91.6034 |
| row · AUC_a micro | 90.1917 | 89.7667 | 86.9626 | 87.0716 |
| row · AP_a micro | 90.3825 | 89.9351 | 87.7081 | 87.5740 |
| row · precision | 68.3575 | 64.2544 | 68.4964 | 68.8406 |
| row · recall | 85.4985 | 88.5196 | 86.7069 | 86.1027 |
| row · F1 | 75.9732 | 74.4600 | 76.5333 | 76.5101 |
| row · FPR | 11.9744 | 14.8995 | 12.0658 | 11.7916 |
| video · precision | 74.4186 | 72.3164 | 72.9885 | 71.4286 |
| video · recall | 91.4286 | 91.4286 | 90.7143 | 89.2857 |
| video · F1 | 82.0513 | 80.7571 | 80.8917 | 79.3651 |
| video · FPR | 17.9592 | 20.0000 | 19.1837 | 20.4082 |
| event · recall | 83.2061 | 87.0229 | 84.7328 | 83.9695 |
| Youden threshold | 0.230227485 | 0.188922532 | 0.210117981 | 0.320431352 |
| judgeable rows | 1425 | 1425 | 1425 | 1425 |
| unjudgeable rows | 255 | 255 | 255 | 255 |
| events | 262 | 262 | 262 | 262 |
| events detected | 218 | 228 | 222 | 220 |
| events never seen, excluded | 0 | 0 | 0 | 0 |
| entries with an event | 140 | 140 | 140 | 140 |
| entries in the macro averages | 95 | 95 | 95 | 95 |
| video TP | 128 | 128 | 127 | 125 |
| video FN | 12 | 12 | 13 | 15 |
| video FP | 44 | 49 | 47 | 50 |
| video TN | 201 | 196 | 198 | 195 |
