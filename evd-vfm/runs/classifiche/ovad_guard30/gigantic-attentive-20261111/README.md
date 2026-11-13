# Gigantic frozen Attentive Probe, seed 42

One epoch (5,524 optimizer steps, effective batch 16), bare frozen V-JEPA 2.1 Gigantic backbone, newly trained four-block Attentive Probe. H100 only. The checkpoint is selected by UCF validation AUROC every 920 steps; all 5,524 training steps are executed. This experiment has no LoRA/VPT prior.

- [UCF validation OVAD](tables/val.md)
- [UCF test OVAD](tables/ucf.md)
- [MIVIA test OVAD](tables/mivia.md)
- [Mean of the two tests](tables/mean_tests.md)

Every table has CSV and JSON counterparts. CSV/JSON use fractions, Markdown percentages. OVAD uses 8 s chunks, guard 30 s and rows up to 32 s. Youden is fitted on UCF validation and transferred unchanged to both tests. Validation metrics describe the calibration data. MIVIA has no validation split. Test averages weight both datasets equally; they are not pooled results or averages over seeds. Protocol Mean is (AUC + AUC_a macro + row F1 + 1 - row FPR + video F1 + event recall) / 6.

The original laboratory evaluator and project export/coverage checks were used, with independent scikit-learn checks. Exact selected step, validation AUROC, checkpoint identity and hashes are in results.json and verification/. The full local package additionally preserves raw scores, config, metadata and OVAD intermediate tables. The shared delivery contains metrics and provenance only.
