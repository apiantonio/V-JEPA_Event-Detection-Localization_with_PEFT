# LoRA to VPT with the inherited trainable head, seed 42

Three VPT epochs (16,572 optimizer steps, effective batch 16), ViT-L with the winning LoRA r32 encoder fused at step 5524 and the head copied from that same step. The head remains trainable together with eight new shallow prompts. H100 only. The checkpoint is selected by UCF validation AUROC every 920 steps; all 16,572 VPT training steps are executed. The LoRA encoder is frozen after fusion; only prompts and the inherited head train. Optimizer moments reset.

- [UCF validation OVAD](tables/val.md)
- [UCF test OVAD](tables/ucf.md)
- [MIVIA test OVAD](tables/mivia.md)
- [Mean of the two tests](tables/mean_tests.md)

Every table has CSV and JSON counterparts. CSV/JSON use fractions, Markdown percentages. OVAD uses 8 s chunks, guard 30 s and rows up to 32 s. Youden is fitted on UCF validation and transferred unchanged to both tests. Validation metrics describe the calibration data. MIVIA has no validation split. Test averages weight both datasets equally; they are not pooled results or averages over seeds. Protocol Mean is (AUC + AUC_a macro + row F1 + 1 - row FPR + video F1 + event recall) / 6.

The original laboratory evaluator and project export/coverage checks were used, with independent scikit-learn checks. Exact selected step, validation AUROC, checkpoint identity and hashes are in results.json and verification/. The full local package additionally preserves raw scores, config, metadata and OVAD intermediate tables. The shared delivery contains metrics and provenance only.
