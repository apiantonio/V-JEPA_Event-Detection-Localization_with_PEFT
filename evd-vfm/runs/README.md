# Run artifacts

A selection of the run outputs the thesis refers to. Backbone weights, the feature store and the
explanation renders are not included. On the UNISA cluster `checkpoint` in the run folders links
to the trained weights (`hpc/README.md`).

| Folder | Content |
|---|---|
| `classifiche/` | Leaderboards of every model on every metric (`classifiche.md`), per class (`per_class.csv`), per video (`per_video.csv`, `per_video_run.csv`), the MIVIA-Crime transfer (`cross_dataset_mivia.csv`), the chunk-ranking summary, the one- vs three-epoch comparison and the notes on each CSV (`LEGGIMI_*.md`) |
| `classifiche/ovad_guard30/classifiche/` | The same models evaluated with the laboratory's OVAD protocol (30 s guard): leaderboards on UCF-Crime and MIVIA-Crime and the cross-dataset table |
| `figures/` | Aggregated comparison figures and the A100/H100 machine comparison |
| `e5b3_efficienza_gpuq/` | Inference efficiency profile (A100, bf16): latency, throughput and memory of every variant, merged and unmerged |
| `statistiche_modelli/` | Cost statistics of the 18 thesis models: training hours per GPU type, selected step over the four-epoch chain, peak memory of the training jobs, response time and alarm delay of the deployed (merged) form |
| `e2c_…_frozen-linear_s42`, `e2d_…_frozen-attentive_s42` | Frozen probes (linear and attentive), seed 42; the attentive probe is the baseline detector of the localization |
| `e3p_…_vptshallow-n8_s42`, `e3p_…_vptdeep-n8_s42`, `e3p_…_vptdeep-n576_s42` | Visual prompt tuning, seed 42 |
| `e3r_…_lora-r32-qkvomlp_s42`, `e3r_…_dora-r32-qkvomlp_s42` | LoRA and DoRA, rank 32, all attention and MLP projections, seed 42 |
| `e3s_…_vptshallow-n8-su-lora-r32_s42` | VPT-shallow n=8 on merged LoRA r=32 ("LoRA+VPT base"), seed 42: the detector of the localization campaigns, on which the class heads B1 and B2 are trained |
| `lcb1_…_s42`, `lcb2_…_s42` | Class heads B1 (branch on the frozen first blocks of the binary head) and B2 (complete class-query head), trained on the frozen encoder of `e3s_…` |
| `lcb3s_…_lora-r32-qkvomlp_s42` → `lcb3s_…_vptshallow-n8-su-lora-r32_s42`, and `lcb3g_…` | B3: a binary and a class head trained with the encoder, losses summed (`lcb3s`) or balanced with GradNorm (`lcb3g`); first stage LoRA, second stage VPT-shallow on the merged LoRA |
| `localization-class-routing-20261021/` | The class-routed localization: the official table, the official evaluations (AP offline and online at 1 fps, FPR of the boxes on the OVAD false-positive rows), thresholds, fit receipts and timing; see its `README.md` |

Each run folder contains:

- `config.yaml` — the resolved configuration;
- `meta.json` — seed, git SHA, protocol hash, numeric regime and environment fingerprint;
- `metrics.json` — every metric of the test set, the threshold and its provenance, the
  budget and the selected step;
- `scores.jsonl` — one score per window, for the test and the validation partitions: the
  source of every per-video number and of the demonstrator replay;
- `history.jsonl` — training loss and validation curves;
- `latenza.json`, `strata.json`, `per_class.csv`, `per_video.csv` — detection latency,
  stratified metrics, per-class and per-video tables (where the run has them);
- figures (`roc.png`, `pr.png`, `confusion.png`, `threshold.png`, `training.png`, …);
- `<run_id>__mivia/` — the same model evaluated on MIVIA-Crime with the threshold
  transferred from UCF-Crime;
- `localization/` (the attentive probe, LoRA+VPT base, DoRA r=32 and the two B3 detectors) —
  the boxes of the automatic localization methods evaluated with that detector, projected
  onto native frames, with the mean and per-class IoU of each; the demonstrator draws them on
  the annotated videos. LoRA+VPT base holds the 163 methods of the first campaign and the
  class-routed configurations of B1 and B2; the B3 detectors their own; the attentive probe
  the methods run on the frozen backbone.

Notes on reading the leaderboards: mean ± sample standard deviation over the seeds
42, 67 and 104; *NA* marks a value that is undefined or not measurable, never a zero; the
detection latency uses the intersection rule and counts detected events only; per-class
metrics compare the positive chunks of one class against all negative chunks.
