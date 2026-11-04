# Tabelle del protocollo OVAD

Sette modelli valutati con il protocollo OVAD sui test di UCF-Crime e MIVIA-Crime.

**Parametri, uguali per tutti i modelli:** chunk size c = 30 s, asking rate h = 4 s, dimensione delle righe T = 32 s, guard g = 30 s. La soglia θ è il punto di Youden (massimo di TPR − FPR) sulle righe giudicabili della validation FS-UCF-Crime finale, calcolata una volta per modello e usata identica su entrambi i test.

**Annotazioni:** FS-UCF-Crime finali per test e validation di UCF-Crime, annotazioni di MIVIA-Crime per il suo test. Gli intervalli in frame sono convertiti in secondi come `[a/fps, (b+1)/fps)`; la durata di ogni video è frame decodificati / fps.

Soglie θ ottenute con Youden sulla validation: UR-DMU 0.9999, PEL4VAD 0.8812, GlanceVAD 0.9245, Holmes-VAU 0.9963, AnomalyCLIP 0.5131, VadCLIP 0.8590, OVadCLIP 0.2384.

Metriche in percentuale. In grassetto il valore migliore di ogni riga (il più basso per le FPR). Con \* le sei metriche che entrano nella media del protocollo.

## UCF-Crime test

| Metrica | UR-DMU | PEL4VAD | GlanceVAD | Holmes-VAU | AnomalyCLIP | VadCLIP | OVadCLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Media del protocollo** | 73.36 | **77.75** | 75.87 | 61.22 | 77.69 | 77.47 | 74.93 |
| Righe · AUC \* | 84.82 | 88.09 | 87.26 | 66.91 | **89.41** | 88.23 | 86.00 |
| Righe · AP | 58.25 | 64.56 | **71.32** | 37.42 | 68.26 | 66.22 | 66.19 |
| Righe · AUC_a macro \* | 75.83 | 81.17 | **85.23** | 77.70 | 82.54 | 77.60 | 78.88 |
| Righe · AP_a macro | 71.56 | 81.11 | **85.46** | 79.64 | 81.87 | 78.70 | 83.48 |
| Righe · AUC_a micro | 70.08 | 71.26 | **75.29** | 66.06 | 71.13 | 66.01 | 70.07 |
| Righe · AP_a micro | 66.65 | 69.29 | **78.78** | 66.27 | 70.11 | 67.51 | 73.35 |
| Righe · Precision | 47.40 | 51.12 | 57.55 | 35.82 | **59.04** | 53.74 | 50.67 |
| Righe · Recall | 79.76 | **89.43** | 72.51 | 63.75 | 74.02 | 84.59 | 80.36 |
| Righe · F1 \* | 59.46 | 65.05 | 64.17 | 45.87 | 65.68 | **65.73** | 62.15 |
| Righe · FPR \* | 27.54 | 26.60 | 16.64 | 35.53 | **15.98** | 22.87 | 24.37 |
| Video · Precision | 63.13 | 58.80 | 64.05 | 43.87 | **67.74** | 65.17 | 62.86 |
| Video · Recall | 80.71 | **90.71** | 70.00 | 66.43 | 75.00 | 82.86 | 78.57 |
| Video · F1 \* | 70.85 | 71.35 | 66.89 | 52.84 | 71.19 | **72.96** | 69.84 |
| Video · FPR | 29.07 | 39.21 | 24.23 | 52.42 | **22.03** | 27.56 | 28.76 |
| Eventi · Recall \* | 76.72 | **87.40** | 68.32 | 59.54 | 73.28 | 83.21 | 77.10 |
| Soglia θ | 0.9999 | 0.8812 | 0.9245 | 0.9963 | 0.5131 | 0.8590 | 0.2384 |
| Righe giudicabili | 1395 | 1395 | 1395 | 1395 | 1395 | 1385 | 1394 |
| Righe non giudicabili | 285 | 285 | 285 | 285 | 285 | 286 | 286 |
| Eventi | 262 | 262 | 262 | 262 | 262 | 262 | 262 |
| Eventi rilevati | 201 | 229 | 179 | 156 | 192 | 218 | 202 |
| Eventi mai visti | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## MIVIA-Crime test

| Metrica | UR-DMU | PEL4VAD | GlanceVAD | Holmes-VAU | AnomalyCLIP | VadCLIP | OVadCLIP |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Media del protocollo** | 68.30 | 71.23 | 66.66 | 58.72 | 74.50 | 69.81 | **77.30** |
| Righe · AUC \* | 77.27 | 76.57 | 80.13 | 61.29 | **85.24** | 82.06 | 84.86 |
| Righe · AP | 64.23 | 63.46 | 72.04 | 50.82 | **78.64** | 74.37 | 76.12 |
| Righe · AUC_a macro \* | 76.03 | 69.91 | 69.22 | 69.38 | 79.29 | 80.75 | **82.72** |
| Righe · AP_a macro | 76.20 | 75.98 | 75.00 | 76.20 | 81.50 | **85.18** | 84.25 |
| Righe · AUC_a micro | 63.99 | 64.69 | 68.18 | 62.72 | 69.42 | 68.40 | **75.84** |
| Righe · AP_a micro | 79.29 | 81.70 | 83.47 | 78.90 | 85.02 | 85.66 | **87.10** |
| Righe · Precision | 66.84 | 63.12 | **72.15** | 48.34 | 68.89 | 71.86 | 70.66 |
| Righe · Recall | 54.89 | 70.64 | 48.51 | 55.74 | 65.96 | 51.06 | **72.77** |
| Righe · F1 \* | 60.28 | 66.67 | 58.02 | 51.78 | 67.39 | 59.70 | **71.70** |
| Righe · FPR \* | 17.25 | 26.15 | **11.86** | 37.74 | 18.87 | 12.77 | 19.19 |
| Video · Precision | 67.57 | 65.58 | **78.05** | 57.50 | 72.09 | 73.00 | 70.42 |
| Video · Recall | 55.56 | **74.81** | 47.41 | 51.11 | 68.89 | 54.07 | 74.07 |
| Video · F1 \* | 60.98 | 69.90 | 58.99 | 54.12 | 70.45 | 62.13 | **72.20** |
| Video · FPR | 21.18 | 31.18 | **10.59** | 30.00 | 21.18 | 16.07 | 24.85 |
| Eventi · Recall \* | 52.50 | 70.50 | 45.50 | 53.50 | 63.50 | 47.00 | **71.50** |
| Soglia θ | 0.9999 | 0.8812 | 0.9245 | 0.9963 | 0.5131 | 0.8590 | 0.2384 |
| Righe giudicabili | 606 | 606 | 606 | 606 | 606 | 603 | 605 |
| Righe non giudicabili | 258 | 258 | 258 | 258 | 258 | 261 | 259 |
| Eventi | 200 | 200 | 200 | 200 | 200 | 200 | 200 |
| Eventi rilevati | 105 | 141 | 91 | 107 | 127 | 94 | 143 |
| Eventi mai visti | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Come leggere le tabelle

- **Media del protocollo:** (AUC + AUC_a macro + F1 righe + (1 − FPR righe) + F1 video + Recall eventi) / 6.
- **Righe:** intervalli lunghi al massimo 32 s fissati dalle annotazioni, uguali per tutti i modelli, interamente dentro o interamente fuori da un'anomalia. Lo score di una riga è il massimo delle finestre che la riguardano.
- **AUC_a macro / micro:** AUC calcolata solo sui video con anomalie; macro = un valore per video e poi la media, micro = tutte le righe di quei video insieme. Lo stesso per AP_a.
- **Video:** ogni video dà fino a due esiti, la rilevazione dell'anomalia (TP/FN) e il silenzio fuori dalle anomalie (FP/TN).
- **Eventi:** le anomalie annotate, unite quando si toccano; un evento è rilevato se almeno una sua riga raggiunge θ.
- **Righe non giudicabili:** righe pulite vicine a un'anomalia in cui ogni finestra da 30 s vede l'anomalia; sono escluse dalle metriche e sono le stesse per tutti i modelli.

## Modelli

- **UR-DMU, PEL4VAD, GlanceVAD, Holmes-VAU:** pesi ufficiali, feature I3D (snippet di 16 frame, center crop). Score della finestra: massimo degli snippet (PEL4VAD: media dei k più alti, k = S/16 + 1).
- **AnomalyCLIP:** pesi ufficiali, CLIP ViT-B/16 su 512 frame ricampionati per finestra, massimo.
- **VadCLIP:** pesi ufficiali, feature CLIP ufficiali, Top-K (K = 16).
- **OVadCLIP:** CALM+Top-K con λtv 0.2, λ1 0.001, K 16.

## Note

- VadCLIP e OVadCLIP hanno poche righe in meno degli altri: le loro feature non coprono gli ultimi secondi di alcuni video.
- In 54 video di MIVIA-Crime l'header dichiara più frame di quelli decodificabili: le durate usano i frame decodificati.
