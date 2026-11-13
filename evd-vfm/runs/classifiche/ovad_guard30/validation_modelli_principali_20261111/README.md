# OVAD validation: main models

Prepared on 11 November 2026 from the existing scores of four V-JEPA 2.1 ViT-L models.

| Results | Tables |
|---|---|
| UCF-Crime validation, seed 42 | [Markdown](validation_s42.md), [CSV](validation_s42.csv) |
| Validation, individual seeds 42, 67 and 104 | [CSV](validation_all_seeds.csv) |
| Mean of the UCF and MIVIA tests, seed 42 | [Markdown](media_test_s42.md), [CSV](media_test_s42.csv) |
| UCF-Crime test, seed 42 | [Markdown](test_ucf_s42.md), [CSV](test_ucf_s42.csv) |
| MIVIA-Crime test, seed 42 | [Markdown](test_mivia_s42.md), [CSV](test_mivia_s42.csv) |

All 15 metrics and the Protocol Mean are included. Tables display percentages; CSV and JSON retain fractions at full precision. `results.json` contains every individual evaluation; `cells/` contains the laboratory evaluator's results and readable reports. Test averages are arithmetic means of the two dataset metrics, with equal dataset weights. They are not averages across seeds. The Protocol Mean is `(AUC + AUC_a macro + row F1 + 1 - row FPR + video F1 + event recall) / 6`.

Models use exactly the versions already reported in the main OVAD comparison: LoRA→VPT is `e3s_ucf_vjepa21l_vptshallow-n8-su-lora-r32` (VPT-shallow on fused LoRA, historical final-state prior; three second-stage cycles); LoRA r32, VPT-shallow n8 and Attentive Probe use their one-epoch runs. These are the existing versions, with no new selection on validation or test. Exact run IDs accompany every record.

OVAD uses chunk size 8 s, guard 30 s, maximum row duration 32 s and the original start grid. Validation uses an 8 s hop; test uses a 4 s hop. Every validation evaluation covers 306 videos and 8,383 chunks. The threshold is the same historical Youden point on judgeable validation rows used on both tests. Validation metrics therefore describe the calibration data and are not an independent test estimate. MIVIA-Crime has no validation split in this experiment.

Inputs, raw score hashes, historical thresholds and test metrics were checked. The unchanged laboratory evaluator was used on CPU; no training or inference was run. Source hashes and check receipts are in `manifest.json` and `verification.json`. The private source exports remain in the original collection; this metrics package contains no checkpoints, videos, ground truth, input scores or private tests.

## Gigantic Attentive Probe availability

No completed frozen Attentive Probe experiment on the Gigantic backbone was found in the local catalogue or the cluster run repository and registry. UCF validation, UCF test, MIVIA test and their average are consequently unavailable for this model. Gigantic LoRA r32 and Gigantic LoRA→VPT are separate available experiments and must not be relabelled as a frozen Probe. The existing Gigantic UCF101 feature-extraction run does not provide event-detection results. The dated availability receipt is `gigantic_attentive_availability.json`.
