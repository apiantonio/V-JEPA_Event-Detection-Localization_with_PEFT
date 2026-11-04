# Binary metrics per annotated class

13 anomalous test classes; one row per class, model and seed. Each run has
per_class.csv and per_class_provenienza.json. The overall CSV concatenates the runs.
Original validation threshold. No training and no new inference.

Chunk: positives of the dominant class of the chunk against all negative test
chunks (ADR-031). The other positive classes are excluded. AUC/AP and P/R/F1 use
the same set. The *_anomalous columns also restrict negatives to anomalous videos.
TP/FN belong to the class; FP/TN and negatives are shared.

Video: videos containing at least one event of the class against all normal
videos. In anomalous videos only an above-threshold chunk intersecting an event of
that class counts; in normal videos any above-threshold chunk does. A video with
events of several classes appears in several rows. The class comes from the
annotated intervals, not from the file or folder name. Per-class diagnostics of
the ADR-047 rule.

Event: strata.json definition (ADR-060). Precision = true alarms of the class /
(true alarms of the class + global false alarms). Recall = detected events /
events of the class. F1 is the harmonic mean. An alarm can touch several classes;
per-class precision/F1 do not add up to the global metrics. Median and p90
latency on detected events only, intersection rule (ADR-061). False alarms do
not become FP events: the denominators are distinct.

NA marks an undefined metric or an unmeasurable latency. Chunk and video precision
without predictions follows the project's 0 convention; event precision without
true or false alarms stays NA, as in strata.json. There is no video (ADR-047) or
event AUC/AP, and no event TN/accuracy. Historical metrics are not rewritten.
Before export, global and per-class AUC/AP, global P/R/F1 and event counts are
checked against metrics.json.
