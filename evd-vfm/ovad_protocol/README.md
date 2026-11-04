# Experimental protocol — evaluation for OVAD

This package computes all the metrics of the protocol from two things you produce: the **model's
anomaly scores on the chunks** of every video, together with the hyperparameters that define them,
and the **labels**, that is, where the anomalies are. The rules of the protocol are explained in
the guide, `GUIDE.pdf` (the same text is in `GUIDE.md`); this file explains how to use the code.

OVAD is anomaly detection: there are no queries, the model looks for any anomaly in the video.
Every video is an entry, and its events are all the annotated anomalies, whatever their class.

Requirements: Python 3.8 or newer and `numpy`. Nothing else.

## What is in here

| File | What it is for |
|---|---|
| `GUIDE.pdf`, `GUIDE.md` | The guide to the protocol: rows, guard, metrics, threshold and how to report results |
| `evaluate_ovad.py` | The evaluation: one label file (triples) and one score file, plus the same for the validation set |
| `protocol_core/` | The computation of rows, threshold and metrics, and the reading and checking of the files |
| `example/` | Synthetic data in the right format and `run_example.sh` to try them |
| `tests/test_protocol.py` | The tests: run them before using the code and after every change |

## First step: the tests

```
python tests/test_protocol.py
```

It must end with `16 of 16 tests passed`.

## Data format

All files are JSON. Every **entry** is a video, identified by its name. The same name must appear in
the labels and in the scores; a video missing on one side is an error, not a skipped entry.

**Labels**

```json
{
  "video_001": {"duration": 73.07, "anomalies": [[16.57, 24.03, "Fighting"], [40.1, 45.0, "Robbery"]]},
  "video_002": {"duration": 120.0, "anomalies": []}
}
```

- `duration`: the duration of the video in seconds.
- `anomalies`: all the anomalies of the video, as triples `[start, end, class]`: `[start, end)` in
  seconds and the class as a string. Empty list for a normal video; the key is required even when
  the list is empty. The class does not enter the metrics: every anomaly is an event. If your
  annotations are in frames with an inclusive end, convert them with `[a/fps, (b+1)/fps)`.

**Scores**

```json
{
  "hyperparameters": {"chunk_size": 8, "asking_rate": 4, "grid": "start"},
  "scores": {
    "video_001": [0.01, 0.02, 0.35, 0.40],
    "video_002": [{"start": 0.0, "end": 8.0, "score": 0.02}]
  }
}
```

- `chunk_size` (*c*): the seconds of video the model reads to produce one anomaly score.
- `asking_rate` (*h*): the seconds between two consecutive queries to the model. The main protocol
  uses 4 s whatever the chunk size; with a different value the code still computes everything but
  reports it among the warnings. It cannot exceed the chunk size, otherwise parts of the video
  would never be seen.
- `grid`: where chunk *j* lies when the scores are a list of numbers.
  - `"start"`: chunk *j* covers `[j·h, j·h + c]`. This is the most common sliding window.
  - `"centered"`: chunk *j* is centered on `(j + ½)·h`.

  In both cases the extent is clipped to `[0, duration]`: the last chunk can be shorter.
- The scores of a video are either a list of numbers, one per chunk in grid order, **or** a list
  of objects with the explicit extent, `start` and `end` in seconds. Use the second form if your
  chunks are cut on frames: at 29.97 fps, for example, the chunk "from 4 to 12 s" actually goes
  from 4.004 to 12.012 s.

Scores can be on any scale: what matters is their order and the threshold.

Validation and test sets have the same format. The chunk size must be the same everywhere, because
the scores come from the same model; the asking rate of the validation set may differ from that of
the test set, and the code reports it.

## How to run it

```
python evaluate_ovad.py \
    --labels test/labels.json --scores test/scores.json \
    --val-labels val/labels.json --val-scores val/scores.json \
    --output results.json
```

From Python the same computation is `evaluate_ovad(labels, scores, val_labels, val_scores)`, with
structures already in memory.

To try it on the example data: `bash example/run_example.sh`.

### The threshold

The thresholded metrics (row F1 and FPR, video level, event recall) need a threshold θ. The
protocol fixes it **on the validation set**: it is the Youden point, that is, the maximum of
TPR − FPR, on the judgeable validation rows. It is taken once and holds for every test set.

If you have no validation set you can pass the threshold by hand with `--threshold 0.35` instead of
the validation files. The result says so among the warnings, because it is not the protocol's
threshold.

### The guard

`--guard` is the guard *g* in seconds. Every chunk gets a guard window: its extent widened by
(g − c)/2 seconds on each side, which for a full chunk is a window of length g centered on the
chunk. If the window overlaps an anomaly, the chunk's score does not count for clean rows. The
guard never moves a score to another row: it leaves it in the row where its center falls, or
discards it.

- If you evaluate a single model you do not need to set it: by default it equals the chunk size,
  and the rule simply becomes "the chunk must not have seen an anomaly".
- If you compare models with different chunk sizes, evaluate **all** of them with the same guard,
  equal to the longest chunk size among them. This way the models are judged as clean on the same
  points of the video, and the short chunk is not penalized near the anomalies, where the long one
  is never judged.

## The results

The script prints a table with the metrics in percent and the counts. With `--output` it writes
everything to JSON, with the metrics as fractions between 0 and 1. A metric that is not defined on
your data, for example because only one class is present, is `null`.

| Key | Metric |
|---|---|
| `mean` | the protocol mean: AUC, AUC_a macro, row F1, 1 − row FPR, video F1, event recall |
| `auc`, `ap` | AUC and AP of the rows, all judgeable rows together |
| `auc_a_macro`, `ap_a_macro` | AUC and AP within every video with anomalies, then the mean over the videos |
| `auc_a_micro`, `ap_a_micro` | AUC and AP of the rows of the videos with anomalies only, all together |
| `row_precision`, `row_recall`, `row_f1`, `row_fpr` | the rows at threshold θ |
| `video_precision`, `video_recall`, `video_f1`, `video_fpr` | the video level at threshold θ |
| `event_recall` | the fraction of anomalies detected at threshold θ: an anomaly is detected if one of its judgeable event rows reaches θ |
| `threshold`, `n_rows`, `n_unjudgeable_rows`, `n_events`, `n_events_detected`, `n_events_unseen`, `n_entries_with_event`, `n_entries_in_macro`, `video_tp`, `video_fn`, `video_fp`, `video_tn`, `hours` | the threshold used and the counts |

The results are under the level `anomalies`; the figure by which models are ranked is its `mean`.
In the counts, `n_entries_with_event` is the number of videos with anomalies and `n_events` the
number of anomalies, after merging those that touch. An anomaly that no chunk overlaps was never
seen by the model: it does not enter `n_events` nor the recall, and is counted in `n_events_unseen`.

When you report the numbers, report next to them the chunk size, the asking rate, the guard, the
threshold θ and the row counts: numbers computed with a different guard or T are not comparable.

## If something goes wrong

- An error starting with `ERROR:` gives the file, the key and the reason: usually a key present on
  one side only, an interval whose end comes before its start, or more scores than the chunks that
  fit in the video with the declared hyperparameters.
- If you change the code, run the tests again: if they fail, the computation is no longer the
  protocol's.
