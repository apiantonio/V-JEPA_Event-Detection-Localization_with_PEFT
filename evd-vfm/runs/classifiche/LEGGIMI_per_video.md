# Per-video leaderboards

`per_video.csv`: one row per video and model, with mean, sample standard
deviation and number of valid seeds for each metric. Anomalous videos only.
`per_video_run.csv`: all runs, one row per video, model and seed.
Each `runs/<run_id>/per_video.csv` holds the same rows for a single run.

score_mediano_chunk_anomali uses all chunks with true label 1 (TP and FN).
score_mediano_chunk_normali uses all those with label 0 (TN and FP).
delta_score_mediano_chunk = anomalous median - normal median. They do not depend
on the threshold. If a set is empty, its median and the delta are NA.
In the aggregated CSV, *_media and *_sd summarise the medians computed per seed;
scores of different runs are never mixed before taking a median.

Each run also has per_video_classi/<Class>.csv, with these three metrics and the
counts for each video of the class, sorted by decreasing delta (NA last). The
class comes from the annotated events: a multi-class video appears in several
files, always with the medians of the whole video. The per-video CSV files keep
the selection of the 140 anomalous videos; the JSON also holds the normal ones.

Test only; validation threshold of the run. Scores from `scores.jsonl`, geometry
from the protocol sampler, events from the lab annotations. Pooled AUC/AP, chunk
counts, video decisions and event counts are checked against `metrics.json`.
The original run receipts stay untouched. Local paths and provenance are
recorded in `classifiche.json`, in the `generato` block.

In the per-run CSV files, anomalous videos are sorted by decreasing chunk F1, AP
and AUC; ties are broken by video_id. `posizione_gruppo` is not a statistical
rank. The aggregated CSV is sorted by video and model, to ease comparison.

AP is 1 when all chunks are positive: the previous export omitted it by using
the AUC guard. AUC needs both labels: without negatives it is NA, as is the
false positive rate. Precision without predictions is 0 by convention, as in the
project metrics; recall and F1 are 0 when no positive is detected.
NA marks an undefined or unmeasurable value, never a zero. The *_stato columns
explain why. *_n counts the valid seeds; an SD needs at least two seeds. Latency
is NA when no event is detected, and its means exclude those seeds: read *_n and
`latenza_stato`.
The JSON also keeps the normal videos to check the full test set: on normal
videos chunk and event precision/recall/F1 are null; counts, accuracy and FPR
remain, as they are defined even without anomalies.
In the CSV files precision/recall/F1 appear only with the chunk_ prefix; the
historical aliases stay in the JSON for compatibility with existing summaries.
Latency uses the intersection rule (ADR-061), in seconds on detected events only:
always read n_eventi_mancati too. It does not include compute time.
Per-video means do not reconstruct the pooled AUC/AP of the whole test set.
