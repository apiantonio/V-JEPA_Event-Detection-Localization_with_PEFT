# OVAD MIVIA-Crime test

Seed 42. Metrics in percent; values in CSV/JSON use fractions.

| Metric | LoRA → VPT-shallow n=8 | LoRA r=32 | VPT-shallow n=8 | Attentive Probe |
|---|---:|---:|---:|---:|
| protocol mean | 86.4898 | 85.4257 | 83.2381 | 82.8192 |
| row · AUC | 92.8545 | 92.7808 | 91.7335 | 91.8368 |
| row · AP | 89.9281 | 89.3946 | 88.1210 | 87.9720 |
| row · AUC_a macro | 92.3958 | 93.1019 | 88.2234 | 88.0035 |
| row · AP_a macro | 94.3427 | 94.6352 | 92.8719 | 92.6301 |
| row · AUC_a micro | 88.4340 | 88.2710 | 86.6954 | 87.0730 |
| row · AP_a micro | 94.6116 | 94.5736 | 93.7325 | 93.9102 |
| row · precision | 85.3774 | 84.5411 | 86.5285 | 83.4171 |
| row · recall | 77.0213 | 74.4681 | 71.0638 | 70.6383 |
| row · F1 | 80.9843 | 79.1855 | 78.0374 | 76.4977 |
| row · FPR | 8.0103 | 8.2687 | 6.7183 | 8.5271 |
| video · precision | 82.7586 | 81.6901 | 82.6772 | 79.6992 |
| video · recall | 88.8889 | 85.9259 | 77.7778 | 78.5185 |
| video · F1 | 85.7143 | 83.7545 | 80.1527 | 79.1045 |
| video · FPR | 13.6612 | 14.2077 | 12.0219 | 14.7541 |
| event · recall | 75.0000 | 72.0000 | 68.0000 | 70.0000 |
| Youden threshold | 0.230227485 | 0.188922532 | 0.210117981 | 0.320431352 |
| judgeable rows | 622 | 622 | 622 | 622 |
| unjudgeable rows | 242 | 242 | 242 | 242 |
| events | 200 | 200 | 200 | 200 |
| events detected | 150 | 144 | 136 | 140 |
| events never seen, excluded | 0 | 0 | 0 | 0 |
| entries with an event | 135 | 135 | 135 | 135 |
| entries in the macro averages | 48 | 48 | 48 | 48 |
| video TP | 120 | 116 | 105 | 106 |
| video FN | 15 | 19 | 30 | 29 |
| video FP | 25 | 26 | 22 | 27 |
| video TN | 158 | 157 | 161 | 156 |
