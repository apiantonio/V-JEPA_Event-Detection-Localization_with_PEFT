# Automatic Event Detection in Surveillance Videos with Video Foundation Models

Code of the master's thesis *Automatic Event Detection in Surveillance Videos with Video
Foundation Models* (University of Salerno, 2026).

The project detects anomalous events in surveillance video with a frozen video foundation
model, **V-JEPA 2.1 ViT-L**, and compares the two families of parameter-efficient
fine-tuning (PEFT) at equal budget: **prompt-based** methods (VPT) and
**reparameterization** methods (LoRA, DoRA). Every score is produced on **causal
windows** that end at the current instant, so the offline numbers describe the online
behaviour of the deployed system. The repository also contains the cross-dataset
evaluation, an efficiency and deployment study, an explainability toolkit, automatic
**spatial localization** of the event — box proposals from VTCD concepts chosen with
Integrated Gradients and routed by the class of the chunk — scored against a manually drawn
ground truth, an interactive demonstrator and a spatial annotation tool.

- **Author:** Antonio Apicella
- **Supervisors:** Prof. Antonio Greco, Prof. Mario Vento (MIVIA Lab, University of Salerno)

---


![End-to-end detection and localization pipeline](media/pipeline.png)

## Demonstrator video

https://github.com/user-attachments/assets/74534ebc-41f0-4f3d-ade6-0285338d5040

**[Watch the original cluster rendering (MP4, 72 seconds)](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/releases/download/demo-v1/demonstrator.mp4).**
FS-UCF-Crime `Shoplifting049`: taking and concealing a phone in a shop. This is the
complete, unmodified movie from the latest existing cluster rendering collection,
`render_best_20261025`, with the updated EVD Studio interface: anomaly score and alarm,
chunk verdict, event and frame IoU, method boxes, dashed reference boxes and timeline.
The model is VPT-shallow on merged LoRA r32; localization uses B2, all-chunk VTCD
dictionaries and the 55% IG-mass rule. The embedded player replaces the static screenshot; the complete movie is also available as a release asset.

This is a qualitative **offline replay**: localization boxes are limited to annotated
intervals and can precede the causal alarm. It is not an online localization timing
measurement. Source identity, checksum and screenshot timestamp are recorded in
[media/demonstrator.json](media/demonstrator.json).

The [pipeline figure](media/pipeline.png) summarizes sampling, PEFT adaptation,
detection and class-routed IG + VTCD localization. Its frames and attribution maps are
schematic illustrations; the demonstrator above shows actual saved model outputs.


## Highlights

- **Reparameterization wins, prompts do not.** DoRA r=32 reaches **0.942 chunk AUC** on
  FS-UCF-Crime (mean of three seeds). At exactly the same number of trainable parameters
  (14.16 M), VPT-deep with 576 prompts falls *below* the frozen attentive probe.
- **Adaptation helps inside anomalous videos.** Over the frozen probe, reparameterization
  adds little chunk AUC but about **+0.04 AP** restricted to anomalous videos: it localizes
  the event within the video rather than separating anomalous from normal videos.
- **Prompts and low-rank updates are largely redundant.** Across the joint and sequential
  combinations of VPT with LoRA/DoRA evaluated on three seeds, none beats DoRA alone beyond
  the seed-to-seed spread, at up to three times its training cost. Longer training does not
  buy accuracy either: at seed 42, three and four epochs in continuation move LoRA and DoRA
  by less than 0.01 chunk AUC, and only the linear probe, under-trained at one epoch, gains.
- **Merging matters for deployment.** Unmerged, LoRA and DoRA are the slowest variants at
  inference (+42.8% and +83.7% latency); once merged into the backbone they cost the same
  as the frozen model. Alarm delay depends strongly on the 4 s window grid;
  direct end-to-end measurements and reconstructed component sums are reported separately.
- **Transfer to an unseen domain.** The same weights, without retraining or threshold
  re-estimation, keep 0.90 chunk AUC on the private MIVIA-Crime test set.
- **Where the event is, without box supervision.** The trained detector becomes a box
  proposer: Integrated Gradients picks the VTCD concepts that carry the anomaly, from the
  dictionary of the class chosen for the chunk. On the official metric (AP@[0.3:0.05:0.7]
  per event, 1 fps, mean of UCF-Crime and MIVIA-Crime) it reaches **25.9 offline AP**,
  against 11–18 for Grad-CAM, IG alone or VTCD concepts alone on the same detector and 23.3
  for the best class-free rule on the frozen baseline, scored against a spatial ground truth
  drawn on 179 test videos.
- **Against the laboratory's reference numbers** (same 8 s protocol): DoRA r=32 obtains
  94.41 / 90.29 chunk AUC on UCF-Crime / MIVIA-Crime, against the best values provided by
  the laboratory, 85.52 (VadCLIP) on UCF-Crime and 81.69 (the laboratory's method) on
  MIVIA-Crime. *The numbers of the other methods come from the laboratory and were not
  reproduced here.*

## Results

FS-UCF-Crime test set (290 videos, 9,116 chunks), protocol EDP-v2. Mean ± sample standard
deviation over seeds 42, 67 and 104; *AUC_a* / *AP_a* are restricted to the 140 anomalous
videos. The full leaderboard (42 models, 100 runs, every metric, per class and per video)
is in [`evd-vfm/runs/classifiche/classifiche.md`](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/blob/main/evd-vfm/runs/classifiche/classifiche.md).

| Metric / attribute | Linear probe (4 ep.) ¹ | Attentive probe (1 ep.) | VPT-shallow n=8 (1 ep.) | VPT-deep n=8 (4 ep.) | VPT-deep n=576 (1 ep.) | LoRA r=32 (3 ep.) | DoRA r=32 (1 ep.) | DoRA r=32 + VPT-shallow n=8 (3 cycles) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Family | frozen | frozen | prompt | prompt | prompt | reparam. | reparam. | joint |
| Trainable adapter params | 0 | 0 | 8,192 | 196,608 | 14,155,776 | 14,155,776 | 14,376,960 | 14,385,152 |
| Chunk AUC | 0.8587 | 0.9310 ± 0.0080 | 0.9360 ± 0.0030 | 0.9325 ± 0.0039 | 0.8882 ± 0.0083 | 0.9386 ± 0.0064 | 0.9422 ± 0.0017 | 0.9405 ± 0.0016 |
| Chunk AP | 0.5333 | 0.7632 ± 0.0092 | 0.7690 ± 0.0081 | 0.7731 ± 0.0094 | 0.6279 ± 0.0201 | 0.7803 ± 0.0186 | 0.7947 ± 0.0037 | 0.7914 ± 0.0037 |
| AUC_a | 0.6807 | 0.8355 | 0.8378 | 0.8409 | 0.7549 | 0.8646 | 0.8669 | 0.8674 |
| AP_a | 0.5732 | 0.7873 | 0.7901 | 0.7970 | 0.6779 | 0.8275 | 0.8305 | 0.8313 |
| Video F1 | 0.8418 | 0.8884 | 0.9068 | 0.8952 | 0.8501 | 0.9099 | 0.9306 | 0.9019 |

The single models trained for one, three and four epochs in continuation are reported once,
in the version with the higher seed-42 chunk AUC on the **test** set: a choice declared in
the leaderboard, not the protocol's validation criterion. One epoch is 5,524 steps.
¹ The fourth epoch of the linear probe was run at seed 42 only, so its row has no spread.
The attentive head (≈ 49.3 M parameters) is trained in every run and is not counted as
adapter.

**Cross-dataset (MIVIA-Crime, 270 videos, seed 42).** The selected weights run on a domain
never seen in training, with the decision threshold transferred from UCF-Crime: DoRA r=32
goes from 0.9441 to 0.9029 chunk AUC, the other models lose 0.02–0.05. MIVIA-Crime has
30.65% positive chunks against 16.36% on UCF-Crime, so its AP is read through the lift.
See section 6 of the leaderboard and `cross_dataset_mivia.csv`.

### Comparison with supplied SOTA: 8-second chunk protocol

Metrics are rows; methods are columns. All values below are percentages, at the Youden
threshold estimated on UCF validation and transferred to MIVIA. Windows are 8 s, hop 4 s.
Thesis results here are **seed 42**, not the three-seed means above. Attentive and DoRA
use one epoch; the sequential models use a one-epoch prior followed by three VPT cycles;
Gigantic uses five LoRA epochs. The supplied methods were evaluated by the laboratory,
not reproduced here. Training supervision and weights are not equally documented for
all supplied methods; these tables do not isolate the effect of the backbone.


**FS-UCF-Crime test — chunk protocol**

| Metric | VadCLIP | Lab method | Ours: Attentive | Ours: DoRA r32 | Ours: LoRA → VPT-s | Ours: DoRA → VPT-s | Ours: Gigantic LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chunk AUC | 85.52 | 84.19 | 93.85 | 94.41 | 94.13 | 94.77 | 94.86 |
| Chunk AP | 49.34 | 56.04 | 77.37 | 79.71 | 79.82 | 80.90 | 81.63 |
| Chunk AUC, anomalous videos | 60.71 | 66.03 | 84.09 | 87.04 | 87.17 | 87.74 | 87.85 |
| Chunk AP, anomalous videos | 51.36 | 60.66 | 79.09 | 83.00 | 83.57 | 83.77 | 84.26 |
| Chunk precision | 41.54 | 34.06 | 60.42 | 58.66 | 49.94 | 60.20 | 58.99 |
| Chunk recall | 75.25 | 85.45 | 79.95 | 82.90 | 88.60 | 83.30 | 84.51 |
| Chunk F1 | 53.53 | 48.70 | 68.82 | 68.70 | 63.88 | 69.89 | 69.48 |
| Event recall ‡ | 84.33 | 87.00 | 89.49 | 92.75 | 96.74 | 92.75 | 90.94 |
| Video precision | 84.25 | 76.07 | 90.97 | 91.84 | 81.55 | 90.48 | 83.65 |
| Video recall | 87.86 | 88.57 | 93.57 | 96.43 | 97.86 | 95.00 | 95.00 |
| Video F1 | 86.01 | 81.85 | 92.25 | 94.08 | 88.96 | 92.68 | 88.96 |


**MIVIA-Crime test — chunk protocol**

| Metric | Lab method | PEL4VAD | GlanceVAD | UR-DMU | Holmes-VAU | VadCLIP | Ours: Attentive | Ours: DoRA r32 | Ours: LoRA → VPT-s | Ours: DoRA → VPT-s | Ours: Gigantic LoRA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chunk AUC | 81.69 | 65.84 | 72.13 | 68.10 | 57.39 | 79.90 | 89.91 | 90.29 | 90.56 | 90.09 | 91.40 |
| Chunk AP | 64.06 | 43.57 | 54.41 | 47.58 | 34.82 | 63.60 | 82.27 | 83.35 | 84.08 | 83.62 | 85.22 |
| Chunk AUC, anomalous videos | 67.45 | 56.41 | 63.26 | 56.77 | 56.53 | 66.08 | 82.35 | 82.93 | 83.64 | 83.66 | 84.16 |
| Chunk AP, anomalous videos | 72.46 | 62.19 | 69.07 | 62.80 | 60.96 | 73.05 | 87.42 | 88.13 | 88.58 | 88.55 | 88.97 |
| Chunk precision | 55.92 | 44.05 | 52.66 | 44.55 | 35.34 | 59.05 | 83.38 | 80.19 | 73.16 | 79.09 | 80.43 |
| Chunk recall | 74.00 | 52.10 | 48.84 | 58.25 | 53.49 | 55.64 | 61.52 | 64.79 | 77.34 | 69.76 | 73.88 |
| Chunk F1 | 63.70 | 47.74 | 50.68 | 50.48 | 42.57 | 57.29 | 70.80 | 71.67 | 75.19 | 74.13 | 77.01 |
| Event recall ‡ | 80.80 | 73.19 | 73.71 | 82.11 | 73.90 | 67.67 | 78.05 | 75.61 | 88.78 | 78.54 | 79.51 |
| Video precision | 71.79 | 68.71 | 72.73 | 66.28 | 56.35 | 72.73 | 87.12 | 85.51 | 80.75 | 83.92 | 86.96 |
| Video recall | 82.96 | 74.81 | 77.04 | 84.44 | 75.56 | 71.11 | 85.19 | 87.41 | 96.30 | 88.89 | 88.89 |
| Video F1 | 76.98 | 71.63 | 74.82 | 74.27 | 64.56 | 71.91 | 86.14 | 86.45 | 87.84 | 86.33 | 87.91 |


Only VadCLIP and the laboratory method were supplied for UCF at 8 s; the other
8-second SOTA results are available only for MIVIA. **‡ Event recall is reported as
supplied, but the laboratory event definition is undocumented in these tables and
need not match EDP-v2; it is not a controlled event-level comparison.**


### Comparison with supplied SOTA: OVAD row protocol

This is a separate protocol: row scores, row/video decisions and event recall are
not the chunk metrics above. All methods use a 4 s hop, rows up to 32 s, a 30 s guard
and UCF-validation Youden thresholds transferred to MIVIA. **Supplied SOTA uses 30 s
input windows; our models use 8 s chunks.** Valid-row populations and supervision
also differ. The comparison is informative, not a matched-window training experiment.

Thesis detectors are at seed 42. Reported continuation versions are selected by the mean
of UCF and MIVIA Protocol Mean on the **test sets**; this is exploratory selection.
In particular, OVAD DoRA and VPT-deep below use four-epoch versions, whereas the
DoRA checkpoint in the 8-second table and the principal weight release is the one-epoch
run. Exact run IDs, unrounded metrics and supplied source tables are in
[comparisons/](comparisons/README.md).


**UCF-Crime test — OVAD rows**

| Metric | UR-DMU | PEL4VAD | GlanceVAD | Holmes-VAU | AnomalyCLIP | VadCLIP | OVadCLIP | Ours: Attentive | Ours: LoRA r32 | Ours: DoRA r32 (4 ep.) | Ours: VPT-s n8 | Ours: VPT-d n8 (4 ep.) | Ours: LoRA → VPT-s | Ours: DoRA → VPT-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Input window (s) | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| Protocol mean | 73.36 | 77.75 | 75.87 | 61.22 | 77.69 | 77.47 | 74.93 | 85.52 | 85.45 | 85.46 | 85.71 | 86.04 | 85.89 | 85.77 |
| Row AUC * | 84.82 | 88.09 | 87.26 | 66.91 | 89.41 | 88.23 | 86.00 | 94.30 | 94.55 | 95.32 | 94.46 | 94.10 | 94.67 | 95.31 |
| Row AP | 58.25 | 64.56 | 71.32 | 37.42 | 68.26 | 66.22 | 66.19 | 85.81 | 86.00 | 88.33 | 85.82 | 85.09 | 86.62 | 87.52 |
| Row AUC_a macro * | 75.83 | 81.17 | 85.23 | 77.70 | 82.54 | 77.60 | 78.88 | 90.79 | 90.79 | 89.97 | 89.71 | 90.59 | 91.39 | 91.67 |
| Row AP_a macro | 71.56 | 81.11 | 85.46 | 79.64 | 81.87 | 78.70 | 83.48 | 91.60 | 91.44 | 90.50 | 90.60 | 92.12 | 91.78 | 92.59 |
| Row AUC_a micro | 70.08 | 71.26 | 75.29 | 66.06 | 71.13 | 66.01 | 70.07 | 87.07 | 89.77 | 90.00 | 86.96 | 87.08 | 90.19 | 90.65 |
| Row AP_a micro | 66.65 | 69.29 | 78.78 | 66.27 | 70.11 | 67.51 | 73.35 | 87.57 | 89.94 | 90.51 | 87.71 | 87.08 | 90.38 | 90.44 |
| Row precision | 47.40 | 51.12 | 57.55 | 35.82 | 59.04 | 53.74 | 50.67 | 68.84 | 64.25 | 67.92 | 68.50 | 69.95 | 68.36 | 64.69 |
| Row recall | 79.76 | 89.43 | 72.51 | 63.75 | 74.02 | 84.59 | 80.36 | 86.10 | 88.52 | 87.01 | 86.71 | 85.80 | 85.50 | 89.12 |
| Row F1 * | 59.46 | 65.05 | 64.17 | 45.87 | 65.68 | 65.73 | 62.15 | 76.51 | 74.46 | 76.29 | 76.53 | 77.07 | 75.97 | 74.97 |
| Row FPR ↓ * | 27.54 | 26.60 | 16.64 | 35.53 | 15.98 | 22.87 | 24.37 | 11.79 | 14.90 | 12.43 | 12.07 | 11.15 | 11.97 | 14.72 |
| Video precision | 63.13 | 58.80 | 64.05 | 43.87 | 67.74 | 65.17 | 62.86 | 71.43 | 72.32 | 70.79 | 72.99 | 73.30 | 74.42 | 71.27 |
| Video recall | 80.71 | 90.71 | 70.00 | 66.43 | 75.00 | 82.86 | 78.57 | 89.29 | 91.43 | 90.00 | 90.71 | 92.14 | 91.43 | 92.14 |
| Video F1 * | 70.85 | 71.35 | 66.89 | 52.84 | 71.19 | 72.96 | 69.84 | 79.37 | 80.76 | 79.25 | 80.89 | 81.65 | 82.05 | 80.37 |
| Video FPR ↓ | 29.07 | 39.21 | 24.23 | 52.42 | 22.03 | 27.56 | 28.76 | 20.41 | 20.00 | 21.22 | 19.18 | 19.18 | 17.96 | 21.22 |
| Event recall * | 76.72 | 87.40 | 68.32 | 59.54 | 73.28 | 83.21 | 77.10 | 83.97 | 87.02 | 84.35 | 84.73 | 83.97 | 83.21 | 87.02 |
| Valid rows (#) | 1395 | 1395 | 1395 | 1395 | 1395 | 1385 | 1394 | 1425 | 1425 | 1425 | 1425 | 1425 | 1425 | 1425 |
| Unjudgeable rows (#) | 285 | 285 | 285 | 285 | 285 | 286 | 286 | 255 | 255 | 255 | 255 | 255 | 255 | 255 |


**MIVIA-Crime test — OVAD rows**

| Metric | UR-DMU | PEL4VAD | GlanceVAD | Holmes-VAU | AnomalyCLIP | VadCLIP | OVadCLIP | Ours: Attentive | Ours: LoRA r32 | Ours: DoRA r32 (4 ep.) | Ours: VPT-s n8 | Ours: VPT-d n8 (4 ep.) | Ours: LoRA → VPT-s | Ours: DoRA → VPT-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Input window (s) | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| Protocol mean | 68.30 | 71.23 | 66.66 | 58.72 | 74.50 | 69.81 | 77.30 | 82.82 | 85.43 | 84.96 | 83.24 | 82.67 | 86.49 | 84.00 |
| Row AUC * | 77.27 | 76.57 | 80.13 | 61.29 | 85.24 | 82.06 | 84.86 | 91.84 | 92.78 | 92.15 | 91.73 | 90.82 | 92.85 | 91.75 |
| Row AP | 64.23 | 63.46 | 72.04 | 50.82 | 78.64 | 74.37 | 76.12 | 87.97 | 89.39 | 89.56 | 88.12 | 86.62 | 89.93 | 88.64 |
| Row AUC_a macro * | 76.03 | 69.91 | 69.22 | 69.38 | 79.29 | 80.75 | 82.72 | 88.00 | 93.10 | 93.30 | 88.22 | 88.11 | 92.40 | 93.34 |
| Row AP_a macro | 76.20 | 75.98 | 75.00 | 76.20 | 81.50 | 85.18 | 84.25 | 92.63 | 94.64 | 95.13 | 92.87 | 92.08 | 94.34 | 94.87 |
| Row AUC_a micro | 63.99 | 64.69 | 68.18 | 62.72 | 69.42 | 68.40 | 75.84 | 87.07 | 88.27 | 88.26 | 86.70 | 87.45 | 88.43 | 87.66 |
| Row AP_a micro | 79.29 | 81.70 | 83.47 | 78.90 | 85.02 | 85.66 | 87.10 | 93.91 | 94.57 | 94.76 | 93.73 | 93.97 | 94.61 | 94.38 |
| Row precision | 66.84 | 63.12 | 72.15 | 48.34 | 68.89 | 71.86 | 70.66 | 83.42 | 84.54 | 83.98 | 86.53 | 87.36 | 85.38 | 80.37 |
| Row recall | 54.89 | 70.64 | 48.51 | 55.74 | 65.96 | 51.06 | 72.77 | 70.64 | 74.47 | 73.62 | 71.06 | 67.66 | 77.02 | 74.89 |
| Row F1 * | 60.28 | 66.67 | 58.02 | 51.78 | 67.39 | 59.70 | 71.70 | 76.50 | 79.19 | 78.46 | 78.04 | 76.26 | 80.98 | 77.53 |
| Row FPR ↓ * | 17.25 | 26.15 | 11.86 | 37.74 | 18.87 | 12.77 | 19.19 | 8.53 | 8.27 | 8.53 | 6.72 | 5.94 | 8.01 | 11.11 |
| Video precision | 67.57 | 65.58 | 78.05 | 57.50 | 72.09 | 73.00 | 70.42 | 79.70 | 81.69 | 81.43 | 82.68 | 84.00 | 82.76 | 76.00 |
| Video recall | 55.56 | 74.81 | 47.41 | 51.11 | 68.89 | 54.07 | 74.07 | 78.52 | 85.93 | 84.44 | 77.78 | 77.78 | 88.89 | 84.44 |
| Video F1 * | 60.98 | 69.90 | 58.99 | 54.12 | 70.45 | 62.13 | 72.20 | 79.10 | 83.75 | 82.91 | 80.15 | 80.77 | 85.71 | 80.00 |
| Video FPR ↓ | 21.18 | 31.18 | 10.59 | 30.00 | 21.18 | 16.07 | 24.85 | 14.75 | 14.21 | 14.21 | 12.02 | 10.93 | 13.66 | 19.67 |
| Event recall * | 52.50 | 70.50 | 45.50 | 53.50 | 63.50 | 47.00 | 71.50 | 70.00 | 72.00 | 71.50 | 68.00 | 66.00 | 75.00 | 72.50 |
| Valid rows (#) | 606 | 606 | 606 | 606 | 606 | 603 | 605 | 622 | 622 | 622 | 622 | 622 | 622 | 622 |
| Unjudgeable rows (#) | 258 | 258 | 258 | 258 | 258 | 261 | 259 | 242 | 242 | 242 | 242 | 242 | 242 | 242 |


\* Protocol Mean is the average of row AUC, anomalous-video macro AUC, row F1,
1 − row FPR, video F1 and event recall. Lower FPR is better; all other displayed
percentage measures are higher-is-better. Macro averages per anomalous video; micro
pools their rows. The supplied SOTA results and their inference settings are preserved
in [comparisons/sources/](comparisons/sources/ovad-supplied.md).

**Efficiency (A100, bf16, one 32-frame window at 384×384, batch 1).**

| Variant | Latency per window | Overhead vs frozen |
|---|---:|---:|
| Frozen backbone + attentive probe | 131.8 ms | — |
| VPT-deep n=576 | — | +1.68% |
| LoRA r=32, unmerged / merged | 188.0 / 132.0 ms | +42.8% / +0.3% |
| DoRA r=32, unmerged / merged | 241.5 / 131.9 ms | +83.7% / +0.3% |

Video decoding adds 40.8 ms per window; every variant runs 14–25× faster than real time
with a 4 s hop. Training hours, the selected step over the four-epoch chain, peak memory
and the response time of the deployed form of the 18 thesis models are in
[`evd-vfm/runs/statistiche_modelli/`](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/blob/main/evd-vfm/runs/statistiche_modelli/statistiche_modelli.md).

## Spatial localization

The detector emits one anomaly score per chunk and no box. To propose *where* the event is,
localization methods run on the selected weights of a trained detector, with no retraining
and no box supervision. The boxes are scored against a spatial ground truth drawn with the
annotation tool on the annotated test events — 140 UCF-Crime videos (276 events) and 39
MIVIA-Crime videos (44 events) — which only the evaluator reads.

### Class-routed localization (official metrics)

For a chunk, Integrated Gradients (IG: 32 Gauss-Legendre nodes, mean of four baselines —
black, noise, mean, blur) attributes the binary logit to the tokens; 3D SLIC splits them into 32 regions; a class is
chosen for the chunk and the regions are assigned to the 8 VTCD concepts of that class's
dictionary; the concepts that carry most of the positive IG mass become boxes on every
temporal plane. The class comes from the dictionaries themselves (NNLS on the 104 concepts
of the 13 stacked dictionaries, no classifier) or from a multiclass head: B1 branches off
the detector's binary head, B2 is a complete class-query head on the frozen encoder, B3
trains a binary and a class head with the encoder (losses summed or balanced with
GradNorm).

The metrics were fixed with the laboratory on 28–30 October 2026: **AP@[0.3:0.05:0.7]**
(main) and AP@[0.5:0.05:0.95] of each event, averaged per class and over the 13 classes, on
one frame per second; **offline** reads every annotated frame from one chunk, **online**
from every chunk holding it, with boxes only where the binary head raises the alarm (the GT
boxes of a missed chunk count as missed); **FPR@TPR90** is the share of frames of the OVAD
false-positive rows (guard 30 s) that still show a box at the threshold keeping a box on
90% of the annotated frames. Mean of UCF-Crime and MIVIA-Crime, %:

| Detector · method | Offline AP | Online AP | FPR@TPR90 ↓ |
|---|---:|---:|---:|
| Frozen attentive probe (baseline) · one global dictionary, one box per concept | 20.57 | 14.61 | 33.82 |
| Frozen attentive probe · best class-free rule | 23.32 | 16.08 | 33.82 |
| LoRA+VPT base · Grad-CAM on the tokens | 11.20 | 8.08 | 51.21 |
| LoRA+VPT base · IG alone (components of every IG level set) | 13.98 | 8.49 | 38.13 |
| LoRA+VPT base · VTCD concepts alone (best of three) | 17.69 | 12.25 | 36.22 |
| LoRA+VPT base · IG + VTCD, class from the dictionaries (NNLS) | 24.70 | 16.34 | 50.07 |
| **LoRA+VPT base · IG + VTCD, class from a head (B1)** | **25.90** | 16.89 | 50.12 |
| B3 (sum) · best rule | 25.02 | **17.57** | 50.18 |

LoRA+VPT base is VPT-shallow n=8 on merged LoRA r=32 (seed 42), the detector of the
localization campaigns. Combining IG with class-routed concepts gains about 10 points of
offline AP over every single-paradigm method; online the gain shrinks, because the boxes
follow the binary alarm, and the best rows show more boxes on the false alarms than the
baseline does. The best rows are chosen on the test set (exploratory); MIVIA-Crime is never
used to choose or estimate anything. Every row, per class, the heads, the ablations and the
state-of-the-art detectors under OVAD are in the official table,
[`evd-vfm/runs/localization-class-routing-20261021/`](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/blob/main/evd-vfm/runs/localization-class-routing-20261021/README.md).

### First comparison: 35 methods

Before the official metrics, 35 methods of six families turned a map — attention, gradients,
Integrated Gradients, VTCD concepts, perturbation estimates, distances from normal memories
or a learned localizer — into boxes on every annotated chunk, scored by one metric: the IoU
between the union of the predicted and of the annotated boxes, per annotated plane, averaged
per event, per class and over the 13 classes. Best method of each family, with LoRA+VPT
base as detector:

| Family | Best method on UCF-Crime | IoU UCF-Crime | IoU MIVIA-Crime | Fit (s) | Inference (s/chunk) |
|---|---|---:|---:|---:|---:|
| Concepts | P07 — VTCD concept chosen by the IG mass of the chunk | 0.322 | 0.311 | 2,099 | 14.4 |
| Integrated Gradients | P13 — IG ensemble, top 5% positive tokens | 0.311 | 0.308 | 0 | 6.7 |
| Perturbations | P42 — LIME on 32 regions | 0.198 | 0.231 | 0 | 8.3 |
| Memories and prototypes | P27 — normal and anomalous prototypes | 0.182 | 0.203 | 493 | 0.5 |
| Learned localizers | P32 — learning from bags of tokens and chunks | 0.166 | 0.189 | 9,404 | 0.4 |
| Gradients and attention | P38 — Grad-CAM adapted to video tokens | 0.161 | 0.182 | 0 | 0.4 |

Inference time includes the decoding and encoding of the chunk, shared by every method.
P07 is flat between 10% and 50% of the IG mass, where it almost always keeps a single
concept. With DoRA r=32 (seed 42) as detector the same methods localize better in 67 of 75
experiments on UCF-Crime: P07 reaches 0.346. The winner is chosen on the test set, so the
comparison is exploratory; MIVIA-Crime is reported and never used to choose.

## Reviewed spatial annotations

The spatial annotations prepared with **Forma** are now included for both test datasets:

| Coverage | FS-UCF-Crime test | MIVIA-Crime test |
|---|---:|---:|
| Reviewed videos | 140 | 39 |
| Completed events | 276 | 44 |
| Reviewed native frames | 131,428 | 9,051 |
| Reviewed frames with no box | 542 | 10 |
| Frame-level box instances | 351,511 | 22,400 |

UCF covers all anomalous test videos; MIVIA covers **39 of 135 anomalous videos**, not
the entire test set. Annotations are manually reviewed, with model-assisted proposals;
this provenance can bias the evaluation. Only the evaluator reads these boxes.

Browse [spatial-annotations/](spatial-annotations/README.md) or download the dataset
archives from the [annotation release](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/releases/tag/annotations-v1).
The format stores zero-based native-pixel inclusive `xyxy` boxes on explicitly reviewed
frame intervals. **Unannotated frames are not negatives**; reviewed empty intervals are
preserved. Each JSON identifies the source video by SHA-256 and native geometry.
The export was expanded and checked frame by frame against a consistent Forma snapshot.
Videos, draft work, personal notes, caches and the working database are not included.

## Method

**Task and protocol (EDP-v2).** The laboratory protocol for FS-UCF-Crime, a fully
supervised version of UCF-Crime with temporal annotations on every video
(train/validation/test: 1,303 / 306 / 290 videos):

- 8 s windows with a 4 s hop on every video, the tail of each video included
  (88,371 training chunks); 32 frames sampled across each window (≈ 4 fps), resized to
  384×384 without cropping (9,216 tokens per window);
- a chunk is positive if it overlaps an annotated event by at least one frame; the model
  emits one anomaly logit per window;
- the decision threshold is the Youden point of the validation ROC and is used for chunk,
  video and event decisions;
- metrics: chunk AUC and AP (primary), AUC_a and AP_a on anomalous videos, precision,
  recall and F1 at chunk, video and event level, and the detection latency.

**Model.** The encoder is V-JEPA 2.1 ViT-L/16 distilled from ViT-G, built from Meta's
source and loaded with `strict=True`; its layer-23 tokens feed an attentive probe (four
cross-attention blocks, 16 heads, one query, one output logit).

**Adaptation methods** (implemented in this repository, without the `peft` library):

- *Reparameterization*: LoRA, rsLoRA and DoRA on the fused QKV projection (separate rank
  per Q/K/V slice), the attention output and the MLP of all 24 blocks, or of a chosen set
  of blocks and projections; mergeable into the backbone weights after training.
- *Prompts*: VPT shallow and deep, implemented as V-JEPA 2.1 registers so that prompts
  receive the identity rotation of the 3D-RoPE, also on a sparse set of blocks; APT and
  DA-VPT are also available.
- *Combinations*: joint training of a reparameterization and a prompt adapter, and two
  sequential regimes — prompts trained on a merged LoRA/DoRA (`preadattamento: fondi`) and
  LoRA/DoRA trained on frozen prompts (`preadattamento: congela`).

**Training.** AdamW, learning rate 1e-4 with cosine schedule, weight decay 0.01, effective
batch 16 (micro-batch 2 × 8 accumulation steps), bf16 autocast, one epoch of 5,524 steps
with model selection on validation AUROC every 920 steps; longer budgets repeat the cosine
cycle in continuation. Training chunks are sampled with weight `1/n_chunk(video)`,
multiplied for positive chunks by their overlap fraction (floor 0.1); augmentation is
photometric only. A resumed training replays the consumed batches without decoding them.
Every run writes its resolved configuration, seed, git SHA, protocol hash and environment
fingerprint next to its metrics and per-window scores.

**Class heads** (for the class-routed localization; detection stays binary). One learned
query per class in a cross-attention block, the class logit read from its own query; B1 and
B2 train on the frozen encoder of the detector for four epochs (B1 on the first two blocks of
its binary head, copied and frozen), B3 trains a binary and a class head with the encoder,
summing the two losses or balancing them with GradNorm. Training chunks are drawn with an
equal share per class, and the selection is the balanced accuracy of validation over the 13
anomaly classes.

## Repository editions and layout

The [public repository](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT) contains the curated delivery; the
[private repository](https://github.com/apiantonio/Tesi-Magistrale_Event-Detection-Localization_V-JEPA2.1_LoRA-VPT) preserves the complete development code and its Git
history, including tests, experimental scripts and design records. Both use this README
and the same project licence. Dataset files, caches, environments and checkpoint binaries
are kept outside Git.

In the public edition the Python project is in `evd-vfm/` and installation metadata and
Windows launchers are at the repository root. In the development edition `src/`, `apps/`,
`scripts/`, `configs/`, `tests/` and `pyproject.toml` are directly at the root.
Commands below run **from the Python project directory**: enter `evd-vfm/` in the public
edition and stay at the root in the development edition.

| Directory | Contents |
|---|---|
| `src/evd/` | Decoding, causal sampling, preprocessing, models, PEFT, training, inference, evaluation and explainability |
| `apps/demo/` | EVD Studio: saved replay, local inference, timeline and localization overlays |
| `apps/annotator/` | Forma spatial annotation tool |
| `configs/` and `manifests/` | Hydra experiments, localization plans and sealed splits |
| `scripts/` | Experiments, export, localization, rendering and reporting entry points |
| `runs/` | Selected score/config/metric artifacts in the public edition; local run storage in development |
| `media/` (repository root) | Pipeline figure and demonstrator preview |
| `weights/` (repository root) | Weight manifest, checksums and download instructions; no binary weights |

## Installation

Tested with Python 3.10 / PyTorch 2.5.1 on Windows and Python 3.12 / PyTorch 2.6.0 on Linux.
Install PyTorch and torchvision for your platform, then from the **repository root**:

```bash
pip install -e ".[demo]"
```

Clone the unmodified [Meta reference implementation](https://github.com/facebookresearch/vjepa2)
at commit `204698b45b3712590f06245fbfba32d3be539812` and set `EVD_VJEPA_DIR` to its directory.
Download the upstream `vjepa2_1_vitl_dist_vitG_384.pt` backbone for ViT-L or
`vjepa2_1_vitG_384.pt` for Gigantic from Meta's release and set `EVD_CKPT_DIR` accordingly.
Upstream source and backbone weights retain their own licence terms.

Set `EVD_RUNS_DIR` to the project `runs/` folder, `EVD_DATA_DIR` to your data root,
`EVD_RAW_DIR` to your videos and `EVD_ANNOT_DIR` to your temporal annotations.
`python -m evd.utils.env` prints the resolved paths. UCF-Crime videos must be obtained
from their authors; FS-UCF-Crime annotations and MIVIA-Crime are supplied by the laboratory.
The GitHub repositories do not redistribute those datasets or third-party temporal annotation datasets.

## Model weights

Principal seed-42 checkpoints are distributed as **GitHub Release assets**, separately
from Git. The [weight manifest](weights/weights-manifest.json) identifies each file,
its SHA-256, selected training step and destination. The [download guide](weights/README.md)
explains installation. Placeholder files in the public run folders mark the destinations.
Weights include the trained head and adapter; the upstream backbone is downloaded separately.

The release covers the attentive probe, LoRA r32, DoRA r32, VPT-shallow n8, VPT-deep n8/n576,
VPT-shallow on LoRA r32 and on DoRA r32, B1/B2 class heads, B3 with summed losses / GradNorm,
and the completed Gigantic LoRA r32 run. An ongoing Gigantic VPT run is not a final model.
Both selected and last head/adapter states are retained because some sequential runs
inherit the last adapter. Optimizer, RNG and training history are omitted: these assets
are for inference, **not for resuming training**. Composite models also require their
source-prior checkpoints and the corresponding run configuration.

## Usage

From the Python project directory:

```bash
# Saved replay: scores and boxes, no model loading or GPU
python apps/demo/serve.py --runs runs --videos /path/to/videos

# Resolve a training configuration without launching training
python scripts/run_experiment.py +experiment=e3r_dora seed=42 --cfg job

# CPU smoke experiment with a synthetic world
python scripts/run_experiment.py +experiment=smoke

# Localization stages and exported demo boxes
python scripts/localization_class_pass.py --help
python scripts/localization_class_fit.py --help
python scripts/localization_class_localize.py --help
python scripts/localization_class_ap.py --help
python scripts/export_demo_localization.py --help
python scripts/render_localization_videos.py --help
```

On Windows, `run_demo.cmd` starts the public delivery; the development launcher is
`Avvia dimostratore.cmd`. Saved replay needs videos and run artifacts. New inference also
needs trained checkpoints and backbone weights; set their paths in the UI. The local
service listens on `127.0.0.1` and uses no CDN.

## Included artifacts and reproducibility

The public delivery retains selected run configurations, metrics, scores, localization
box exports, the localization comparison table without B4, leaderboards and model cost
statistics. Training checkpoints are release assets. The development repository contains
the complete source and tests; its ignored local `runs/` is not automatically committed.
Private tests, working annotation databases, environments, intermediate archives, video datasets and restricted
RBDC code are excluded from the public repository. AP-only evaluation remains available
with `--ap-only`; full spatial evaluation can use the reviewed spatial annotations included in this repository.

Every run records its configuration, seed, Git SHA, protocol hash and environment.
Means and standard deviations use seeds 42, 67 and 104 where stated; localization uses
seed 42. A spread over seeds is not a confidence interval. Model selection and detection
thresholds use validation data; test-based choices in the displayed leaderboards and
localization rankings are explicitly exploratory. The laboratory split is retained,
including its known cross-partition duplicate videos. OVAD metrics and EDP-v2 metrics
use different rules and are not interchangeable.

## License

Project code is available under the **PolyForm Noncommercial License 1.0.0**; see
[LICENSE](LICENSE) for the full terms. Required notice: Copyright 2026 Antonio Apicella.
Third-party code, pretrained backbones and datasets retain their respective licences;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The project licence does not replace
those terms or grant rights to the source surveillance footage.

## Acknowledgements and references

- V-JEPA 2 / 2.1: M. Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable
  Understanding, Prediction and Planning*, 2025 — code and weights from
  [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2).
- LoRA: E. J. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022.
- DoRA: S.-Y. Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation*, ICML 2024.
- VPT: M. Jia et al., *Visual Prompt Tuning*, ECCV 2022.
- VTCD: M. Kowal et al., *Understanding Video Transformers via Universal Concept
  Discovery*, CVPR 2024.
- Integrated Gradients: M. Sundararajan, A. Taly, Q. Yan, *Axiomatic Attribution for Deep
  Networks*, ICML 2017.
- Grad-CAM: R. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*, ICCV 2017.
- GradNorm: Z. Chen et al., *GradNorm: Gradient Normalization for Adaptive Loss Balancing in
  Deep Multitask Networks*, ICML 2018.
- SLIC: R. Achanta et al., *SLIC Superpixels Compared to State-of-the-Art Superpixel
  Methods*, TPAMI 2012.
- COCO AP: T.-Y. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014;
  Soft-NMS: N. Bodla et al., ICCV 2017; Weighted Boxes Fusion: R. Solovyev et al., Image
  and Vision Computing 2021.
- RBDC/RBDR: B. Ramachandra, M. Jones, *Street Scene: A New Dataset and Evaluation Protocol
  for Video Anomaly Detection*, WACV 2020.
- UCF-Crime: W. Sultani, C. Chen, M. Shah, *Real-world Anomaly Detection in Surveillance
  Videos*, CVPR 2018.
- FS-UCF-Crime annotations, MIVIA-Crime and the OVAD evaluation code: MIVIA Lab,
  University of Salerno (Carletti, Greco, Marseglia, Vento), *Online Fully Supervised
  Video Anomaly Detection* (under review).

The laboratory data and code are used under the terms agreed with the MIVIA Lab and are not
redistributed with a public licence.

**Third-party code in `evd-vfm/src/evd/vendor/`**, each file naming its source, commit and
changes: the V-JEPA 2.1 transforms and RoPE attention (Meta, MIT; the RandAugment and random
erasing parts derive from `timm`, Apache 2.0, original headers kept), the COCO matching and
AP accumulation of `pycocotools` 2.0.11 (BSD-2-Clause), the NMS and Soft-NMS of Bodla et al.
(MIT) and Weighted Boxes Fusion (MIT). Their notices are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
