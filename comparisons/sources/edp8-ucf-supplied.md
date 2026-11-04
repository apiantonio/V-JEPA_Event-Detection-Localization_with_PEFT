# Protocollo 8 s — UCF-Crime test: Proposed vs VadCLIP

Finestre da 8 s, passo 4 s. Soglie calibrate sul validation set di FS-UCF-Crime.

## Metriche senza soglia (chunk-level)

| Metodo | AUC | AP | AUC_a | AP_a |
|---|---|---|---|---|
| VadCLIP | **85.52** | 49.34 | 60.71 | 51.36 |
| Proposed | 84.19 | **56.04** | **66.03** | **60.66** |

## Soglia Youden (protocol) — VadCLIP 0.7069, Proposed 0.0054

| Metodo | P | R | F1 | BA | P_a | F1_a | R_evt | video P | video R | video F1 | video BA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VadCLIP | 41.54 | 75.25 | **53.53** | **77.27** | 44.28 | 55.75 | 84.33 | 84.25 | 87.86 | **86.01** | **86.26** |
| Proposed | 34.06 | 85.45 | 48.70 | 76.55 | 42.81 | **57.04** | **87.00** | 76.07 | 88.57 | 81.85 | 81.29 |

## Soglia F1-max — VadCLIP 0.9738, Proposed 0.0605

| Metodo | P | R | F1 | BA | P_a | F1_a | R_evt | video P | video R | video F1 | video BA |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VadCLIP | 49.68 | 46.55 | 48.06 | 68.66 | 49.75 | 48.09 | 58.18 | 97.85 | 65.00 | 78.11 | 81.83 |
| Proposed | 42.17 | 69.35 | **52.45** | **75.38** | 47.21 | **56.18** | **77.45** | 86.15 | 80.00 | **82.96** | **84.00** |
