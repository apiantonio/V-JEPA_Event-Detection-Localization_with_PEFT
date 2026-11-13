# OVAD validation UCF-Crime

Seed 42. Metrics in percent; values in CSV/JSON use fractions.

| Metric | LoRA → VPT-shallow n=8 | LoRA r=32 | VPT-shallow n=8 | Attentive Probe |
|---|---:|---:|---:|---:|
| protocol mean | 84.0959 | 84.5350 | 82.5431 | 83.0761 |
| row · AUC | 93.9485 | 94.5676 | 93.4495 | 93.8459 |
| row · AP | 82.2600 | 84.6070 | 81.6044 | 83.2479 |
| row · AUC_a macro | 89.4858 | 89.3949 | 85.1747 | 89.0782 |
| row · AP_a macro | 86.1067 | 85.2842 | 84.2626 | 86.6200 |
| row · AUC_a micro | 85.9390 | 87.5599 | 84.7514 | 85.4251 |
| row · AP_a micro | 84.8256 | 86.8372 | 84.1747 | 85.9182 |
| row · precision | 62.0553 | 64.6939 | 59.0317 | 58.8454 |
| row · recall | 83.9572 | 84.7594 | 84.7594 | 84.4920 |
| row · F1 | 71.3636 | 73.3796 | 69.5939 | 69.3743 |
| row · FPR | 9.9430 | 8.9591 | 11.3931 | 11.4448 |
| video · precision | 74.8466 | 71.5116 | 71.1765 | 73.2919 |
| video · recall | 86.5248 | 87.2340 | 85.8156 | 83.6879 |
| video · F1 | 80.2632 | 78.5942 | 77.8135 | 78.1457 |
| video · FPR | 16.5323 | 19.7581 | 19.7581 | 17.3387 |
| event · recall | 79.4574 | 80.2326 | 80.6202 | 79.4574 |
| Youden threshold | 0.230227485 | 0.188922532 | 0.210117981 | 0.320431352 |
| judgeable rows | 2305 | 2305 | 2305 | 2305 |
| unjudgeable rows | 251 | 251 | 251 | 251 |
| events | 258 | 258 | 258 | 258 |
| events detected | 205 | 207 | 208 | 205 |
| events never seen, excluded | 0 | 0 | 0 | 0 |
| entries with an event | 141 | 141 | 141 | 141 |
| entries in the macro averages | 83 | 83 | 83 | 83 |
| video TP | 122 | 123 | 121 | 118 |
| video FN | 19 | 18 | 20 | 23 |
| video FP | 41 | 49 | 49 | 43 |
| video TN | 207 | 199 | 199 | 205 |
