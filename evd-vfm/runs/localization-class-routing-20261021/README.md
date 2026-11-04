# Class-routed localization

Campaign `localization-class-routing-20261021` (21 October – 3 November 2026): where the
anomalous event is inside a chunk, as boxes, with no box supervision. The detector scores the
chunk; Integrated Gradients (IG) attributes its binary logit to the tokens; 3D SLIC splits the
tokens into 32 regions; a class is chosen for the chunk and the 8 VTCD concepts of that
class's dictionary are assigned to the regions; the concepts carrying most of the positive IG
mass become boxes on every temporal plane, in native video coordinates.

The page [`tabella_localizzazione.html`](tabella_localizzazione.html) holds every result: one
self-contained file (open it offline), the same data in
[`tabella_localizzazione.json`](tabella_localizzazione.json).

## Detectors and class heads

| Role | Run | What it is |
|---|---|---|
| Baseline | `e2d_ucf_vjepa21l_frozen-attentive_s42` | frozen V-JEPA 2.1-L and attentive binary head; class-free configurations only |
| LoRA+VPT base ("ours") | `e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42` | VPT-shallow n=8 on merged LoRA r=32, binary head |
| B1 · MultiClass-Branch | `lcb1_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42` | class head branching off the first two blocks of the detector's binary head, encoder frozen |
| B2 · MultiClass-Full | `lcb2_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42` | complete class-query head (14 queries) on the frozen encoder of the detector |
| B3 · MultiClass-Own, sum | `lcb3s_ucf_vjepa21l_lora-r32-qkvomlp_s42` → `lcb3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42` | binary and class heads trained with the encoder, losses summed (LoRA one epoch, then VPT-shallow three) |
| B3 · MultiClass-Own, GradNorm | `lcb3g_ucf_vjepa21l_lora-r32-qkvomlp_s42` → `lcb3g_ucf_vjepa21l_vptshallow-n8-su-lora-r32_s42` | the same with GradNorm loss balancing |

The run folders are in `runs/` next to this one; on the cluster `checkpoint` links to the
weights. The four detectors carry
`localization/`, the boxes the demonstrator draws (no spatial ground truth inside).

**Who chooses the class → which dictionary.** `No class→Concepts-Global` uses one dictionary
for every class (P07). `NNLS-<rule>` chooses the class without a classifier, decomposing every
region on the 104 concepts of the 13 stacked class dictionaries (rules RawSum, RegionOneVote,
TokenOneVote, NetVote). `MultiClass-Branch`, `MultiClass-Full` and `MultiClass-Own` are the
heads B1, B2 and B3. `Concepts-References` are the class dictionaries learned on 16
reference chunks per class (A1), `Concepts-AllChunks` those learned on every training chunk
inside an event of the class (A2). The oracle takes the annotated class and is an upper
bound.

## Official metrics

Decided on 28–30 October 2026:

- **AP@[0.3:0.05:0.7]** (main) and **AP@[0.5:0.05:0.95]** (COCO): COCO-style AP of each event
  over its images, averaged over the IoU thresholds, then over the events of a class, then
  over the 13 classes; the ground truth of an image is the union of its boxes.
- **1 fps**: the native frame nearest to every second of the video.
- **Offline**: every annotated frame read from one chunk (the one that ends first).
  **Online**: every chunk holding the frame, boxes only on the chunks the binary head flags
  at the OVAD threshold; the boxes of an unflagged chunk count as missed.
- **FPR_box@TPR90**: share of the 1 fps frames of the OVAD false-positive rows (guard 30 s)
  that still show a box at the threshold keeping a box on 90% of the annotated frames of the
  true-positive rows. Lower is better.
- IG with 32 Gauss-Legendre nodes (256 in the first passes; the page keeps both).

## Results

Mean of FS-UCF-Crime and MIVIA-Crime, %, from the page. "Best" is the row with the highest
offline main AP averaged over the two datasets, oracle left out: **chosen on the test set,
exploratory**.

| Detector · step | Configuration · box variant · confidence | Offline AP | Online AP | FPR@TPR90 |
|---|---|---:|---:|---:|
| Baseline · simple | No class→Concepts-Global · concept box · IG mass | 20.57 | 14.61 | 33.82 |
| Baseline · best without class | No class→Concepts-Global · largest 8-conn. component · IG mass | 23.32 | 16.08 | 33.82 |
| Ours · simple | No class→Concepts-Global · concept box · IG mass | 21.18 | 15.24 | 40.24 |
| Ours · best | MultiClass-Branch→Concepts-AllChunks · largest 4-conn. component · IG share | **25.90** | 16.89 | 50.12 |
| B3 sum · best | No class→Concepts-Global · union of 4-conn. components · IG share | 25.02 | **17.57** | 50.18 |
| B3 GradNorm · best | MultiClass-Own→Concepts-AllChunks · largest 8-conn. component · IG share | 23.59 | 16.80 | 61.04 |

**Other paradigms** (approved on 29 October 2026), on the LoRA+VPT base detector and the same
chunks:

| Paradigm | Method | Offline AP | Online AP | FPR@TPR90 | s / chunk |
|---|---|---:|---:|---:|---:|
| Neither IG nor concepts | Grad-CAM on the tokens | 11.20 | 8.08 | 51.21 | 0.43 |
| IG alone | 8-connected components of every IG level set, NMS 0.5 | 13.98 | 8.49 | 38.13 | 3.36 |
| Concepts alone | VTCD with the official CRIS | 15.52 | 11.23 | 49.75 | 2.59 |
| Concepts alone | VTCD, global dictionary | 16.32 | 11.15 | 49.51 | 4.61 |
| Concepts alone | VTCD, class from MultiClass-Branch → its dictionary | 17.69 | 12.25 | 36.22 | 4.62 |
| IG + VTCD | ours, class from a head | **25.90** | **16.89** | 50.12 | 7.35 |
| IG + VTCD | ours, class from the dictionaries (NNLS, no head) | 24.70 | 16.34 | 50.07 | 7.34 |

Seconds: median per chunk on an H100, decoding and encoding included, IG at 32 nodes.

**Heads.** Binary heads, OVAD Protocol Mean (guard 30 s) on FS-UCF-Crime / MIVIA-Crime:
LoRA+VPT base 85.89 / 86.49, baseline 85.52 / 82.82, B3 sum 84.31 / 82.80, B3 GradNorm
83.79 / 84.07. Class heads, accuracy per event then class on the annotated test chunks:
B1 28.68 / 25.42, B2 36.05 / 33.52, B3 sum 38.63 / 34.99, B3 GradNorm 34.47 / 38.92.

Reading them: combining IG with class-routed VTCD concepts gains about 10 points of offline
AP over every single-paradigm method and about 3–5 over the class-free rule of the same
detector; online the gain halves, because the binary head misses some annotated chunks, and
the false-positive rate of the best rows is higher than the baseline's.

## Contents

| Path | Content |
|---|---|
| `tabella_localizzazione.html`, `.json` | the official table: every row (detector, configuration, box variant, confidence), per class, the heads, the baseline-to-ours ladder, the comparisons and their ablations, the IG node sweep, the state-of-the-art detectors under OVAD |
| `evaluation/f1-n32-{offline,online}-<stage>/` | the official evaluations (1 fps, 32 IG nodes): `summary.json`, `summary_by_threshold.csv`, per-event AP at every IoU threshold (`metrics_per_event_<dataset>.npz` with `axes_<dataset>.json`), and the frame counts of the FPR (`frame_fpr_<dataset>.json`) |
| `detection/ovad_guard30/risultati.json` | OVAD metrics of the binary heads of the detectors |
| `calibration/<detector>/thresholds.json` | mass fractions of the calibrated threshold (the share of validation positives keeping one concept) |
| `fits/<detector>/fit_receipts.json` | receipts of the dictionary fits |
| `timing/timing_report.json`, `timing/training_hours.json` | the cost of every step per chunk on an H100, and the training hours of the heads |
| `ig_convergence*.json` | completeness of IG per detector |

Stages: `campaign` (LoRA+VPT base), `b3s`, `b3g`, `attentive` (baseline), `cmp-ig` and
`cmp-other` (the comparison methods).

Inputs of the evaluations, private (the ground truth of the test events drawn with the annotation
tool, and the MIVIA-Crime set):

| Path | Content |
|---|---|
| `gt-snapshot/` | the frozen spatial ground truth that only the evaluator reads |
| `plan/` | chunk lists, intervals and OVAD rows of the detectors |
| `fits/<detector>/` | the VTCD dictionaries |
| `passes/`, `ig-nodes/n32/`, `ig-nodes-attentive/n32/` | IG maps (256 and 32 nodes), SLIC regions and class logits |
| `stages/` | the sealed predictions of the six official stages |

The last two rows are 30 GB on the cluster; the local copy keeps them as the archives
`archives/stages-official.tar.gz` and `archives/ig-maps.tar.gz`.

Not included: the RBDC/RBDR evaluations (their reference script is under a non-commercial,
no-derivatives licence), and the separate class classifiers B4, left out of every comparison.

The scripts are `scripts/localization_class_*.py` and `scripts/localization_compare*.py`;
the plan is `configs/localization_class_routing_20261021.json`. MIVIA-Crime is test only:
dictionaries, thresholds and head thresholds come from FS-UCF-Crime.
