# Confronto tra 1 e 3 epoche sui due test set

> ⚠️ **Superato il 25 settembre, sera.** La versione riportata nelle classifiche non segue la colonna "da riportare" qui sotto: è quella di `VERSIONE_RIPORTATA` in `scripts/classifiche_modelli.py` — tre epoche per Linear probe, Attentive probe e LoRA r=32, una per VPT-shallow n=8, VPT-deep n=8 e DoRA r=32 (chunk-AUC di test FS-UCF-Crime al seme 42; l'attentive a tre epoche per decisione). Vedi `docs/10` §E3-M, postilla del 25 settembre.


Sei coppie E3-M, seme 42. Ogni cella mostra **1 epoca → 3 epoche**; valori in percentuale.
La colonna "da riportare" sceglie la maggiore AUC media aritmetica fra FS-UCF-Crime e MIVIA-Crime.

| Metodo | AUC UCF | AUC MIVIA | AUC media | Video F1 media | Da riportare per AUC |
|---|---:|---:|---:|---:|---|
| Linear probe | 84,73 → 85,62 | 79,33 → 81,17 | 82,03 → **83,40** | 77,26 → 79,73 | **3 epoche** |
| Attentive probe | 93,85 → 93,39 | 89,91 → 88,53 | **91,88** → 90,96 | 89,20 → 89,75 | **1 epoca** |
| VPT-shallow n=8 | 93,94 → 93,47 | 89,82 → 89,93 | **91,88** → 91,70 | 88,64 → 87,91 | **1 epoca** |
| VPT-deep n=8 | 93,44 → 93,11 | 89,31 → 88,15 | **91,38** → 90,63 | 87,25 → 86,27 | **1 epoca** |
| LoRA r=32 | 93,93 → 94,51 | 90,74 → 90,02 | **92,34** → 92,26 | 90,74 → 88,68 | **1 epoca** |
| DoRA r=32 | 94,41 → 94,11 | 90,29 → 89,74 | **92,35** → 91,93 | 90,26 → 89,34 | **1 epoca** |

Per Video F1 media, Attentive probe preferisce 3 epoche; negli altri cinque confronti la scelta coincide con quella per AUC media.
VPT-shallow n=8 e LoRA r=32 hanno differenze di AUC media rispettivamente di 0,18 e 0,08 punti percentuali a favore di 1 epoca; i due test set non concordano per queste due coppie.

Fonte: `cross_dataset_mivia.csv`, campi `chunk_auc_ucf_stessi_semi`, `chunk_auc_mivia`, `video_f1_ucf_stessi_semi` e `video_f1_mivia`. La Video F1 su MIVIA-Crime usa la soglia scelta sulla validazione FS-UCF-Crime e trasferita senza ricalcolo.
