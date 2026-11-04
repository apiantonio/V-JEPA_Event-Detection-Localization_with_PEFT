# OVAD experimental protocol

OVAD is an anomaly detection problem: the model receives a video chunk and gives an anomaly score, that is, how likely it is that an anomaly of any kind is happening in the chunk. There are no queries: every video comes with the temporal annotations of all the anomalies it contains, each labelled with its class.

The metrics of the experimental protocol are not computed directly on the chunks but on time intervals fixed by the annotation, called rows, and then on videos and events: this makes models that use different chunk sizes much more comparable.

Together with this guide comes the code that computes all the metrics from the chunk scores and the labels.

Below we define a few fixed parameters that are an important part of the protocol.

## The constants

The protocol fixes:

- the **asking rate**: the shift between the start of one chunk and the start of the next, that is, how often (in seconds) the model is queried;
- the **row size**: it is the same for all models and is the basic unit of the evaluation protocol.

The table below lists the parameters of the experimental protocol:

| Symbol | What it is | Value | Who sets it |
| --- | --- | --- | --- |
| c | chunk size: the seconds of video the model reads to produce one anomaly score | free, declared | the model |
| h | asking rate: the seconds between two consecutive queries to the model | 4 s, whatever c | the protocol |
| T | row size: maximum length of a row, the resolution at which the model is judged | 32 s | the protocol |
| g | guard: the window, centered on the chunk, used to decide whether the chunk's score can count as evidence on clean footage | equal to c; the longest c when models with different chunk sizes are compared | the protocol, declared per table |
| θ | threshold of the thresholded metrics | Youden point on the validation rows | dynamic, computed on the data |

A few additional notes:

- The asking rate cannot exceed the chunk size, otherwise parts of the video would never be seen. Keeping it fixed at 4 s makes the density of decisions over time independent of the chosen chunk.
- There is no tolerance on the annotations: an overlap counts if it is greater than zero, at every level.
- The guard equals the chunk size when a single model is evaluated, and in that case it requires no choice. Its role when models with different chunk sizes are compared is explained in its own section.

## Chunks and rows

**What a row is.** The protocol is built around a basic unit, the row. A row is a time interval of the video on which the protocol expresses a judgment: in that interval an anomaly either occurs or it does not. Rows have two properties, and the whole protocol rests on them:

- they are **pure**: a row lies entirely inside an anomaly, and is called an event row (label 1), or entirely outside, and is called a clean row (label 0). There are no half rows, so every row has an unambiguous label;
- they **do not depend on the model**: they are built only from the annotation and from T (the row size), without using chunk size, asking rate or scores. Any two models are therefore judged on exactly the same intervals, with the same labels.

This is precisely why the chunks are not judged directly: chunks depend on the model. A model with 32 s chunks and one with 8 s chunks produce different chunks, in different numbers, and a chunk that straddles an anomaly is neither fully positive nor fully negative. On the rows, instead, all models answer the same questions.

**How rows are built.** In OVAD the unit of evaluation is the video. Since the model looks for any anomaly, the events of a video are all its annotated anomalies, whatever their class; if two anomalies touch or overlap they are merged into a single event. The class therefore does not enter the metrics, and a normal video has no events. The timeline is cut at the event boundaries, and every resulting segment is split into rows. A segment L seconds long is split into k = ⌈L/T⌉ rows, where ⌈·⌉ means rounding up (for example ⌈49/32⌉ = ⌈1.53⌉ = 2): this is the smallest number of rows that are at most T long. The k rows all have the same length L/k: the segment is not cut into blocks of T seconds leaving a remainder at the end, but divided into equal parts, so that no row is much shorter than the others. The timeline of the video ends where the last chunk ends.

For example, a 73 s video with an anomaly from 16.5 to 24 s produces four rows: [0, 16.5] clean, [16.5, 24] event, [24, 48.5] and [48.5, 73] clean.

T therefore sets the resolution at which the model is judged: a long clean stretch is split into several rows, and each of them can be a false alarm of its own.

**From chunk scores to row scores.** The model does not produce a score per row, but an anomaly score per chunk. These scores are brought onto the rows as follows:

- the model is queried every h seconds;
- each time it reads c seconds of video and produces an anomaly score for that chunk. The interval of video it has read is the chunk's **extent**, and the midpoint of the extent is the chunk's **center**. The last chunk can be cut by the end of the video, and its extent is then shorter than c;
- the score is assigned to the rows with the following rule:
    - for an **event row**, all the chunks whose extent overlaps the row (which lies inside an anomaly) count, wherever their center falls: a chunk that saw even just part of the anomaly is evidence on the anomaly;
    - for a **clean row**, the chunks whose center falls in the row count, provided they did not see an anomaly, that is, provided their extent does not overlap any anomaly. This way a chunk that saw an anomaly is never counted as a false alarm.

The score of a row is then the maximum of the scores it has collected: one chunk that responds is enough for the model to have signaled something in that interval.

It remains to say what happens when a row has no candidates:

- for a clean row it can happen that no center falls in it, when the row is shorter than the asking rate, for example a short stretch between two anomalies. In that case the chunk whose center is nearest to the row is used, under the same condition of not having seen an anomaly;
- for an event row, instead, there is no fallback: if no chunk overlaps it, any other chunk would not have seen the anomaly, and using its score would mean judging the anomaly on clean footage. That row therefore becomes unjudgeable. This never happens with a regular grid and h ≤ c, because the extents cover the whole timeline.

Going back to the example, with 8 s chunks queried every 4 s: the event row 16.5–24 takes the maximum of chunks 12–20, 16–24 and 20–28, the only ones whose extent overlaps it. The clean row 0–16.5 takes the maximum of chunks 0–8, 4–12 and 8–16; chunk 12–20 has its center at 16 s, inside the row, but it saw the start of the anomaly and therefore does not count.

**Unjudgeable rows.** A row is unjudgeable when no chunk brings valid evidence on it according to the rules above: a clean row in which all the candidate chunks saw an anomaly, or an event row that no chunk overlaps. An unjudgeable row is left out of the row metrics and is reported in the count of unjudgeable rows.

## The guard

The rules of the previous section are enough when a single model is evaluated. When models with different chunk sizes are compared, however, there is a problem that the rules alone do not solve, and this is why the protocol has one more parameter, the guard.

**The problem.** The condition that excludes a chunk from the clean rows, "the chunk must not have seen an anomaly", depends on the length of the chunk. A long chunk sees farther, so it is excluded from the clean rows in a wide band around every anomaly; a short chunk is excluded only in a narrow band. The rows are the same for all models, but the points of the video at which a model is judged as a possible false alarm are not.

Consider an anomaly from 16.6 to 24 s and two models, one with 8 s chunks and one with 32 s chunks, both with a chunk centered at 12 s:

| Model | Extent of the chunk centered at 12 s | Does the extent overlap the anomaly? | Where the score goes |
| --- | --- | --- | --- |
| 8 s chunks | 8–16 s | no | into the clean row 0–16.6 |
| 32 s chunks | 0–28 s | yes | into the event row |

The 8 s model is judged as a possible false alarm at 12 s, that is, a few seconds before the anomaly starts: exactly where it is easiest to make a mistake, because the footage already looks like the anomaly. The 32 s model, at the same point, is never judged on clean footage, and its score ends up in the event row instead. The comparison thus ends up favoring the long chunk without the model being actually better. The effect is not small: on the real scores of an 8 s model, excluding its chunks from the clean rows with the band of a 32 s chunk instead of its own raises the protocol mean by about 2–3.6 points.

**What the guard is.** The guard g makes the band around the anomalies in which scores do not count for the clean rows the same for all models:

- every chunk gets a guard window: its extent widened by (g − c)/2 seconds on each side, which for a full chunk is a window of length g centered on the chunk's center;
- if this window overlaps an anomaly, the chunk's score is left out of the clean rows, because otherwise models with different chunk sizes would not be comparable. The score still counts for the event rows that its extent overlaps.

**Which value to use.**

- When a single model is evaluated, g = c: the window coincides with the chunk's extent and the rule is exactly the one of the previous section. This is the default value and requires no choice.
- When models with different chunk sizes are compared, g is the longest chunk size among those compared, and it holds for all models. In the example, with g = 32 s the window of the 8 s model centered at 12 s becomes 0–28 s, overlaps the anomaly, and its score is discarded, as happens to the 32 s model: from then on the two models are judged as clean on the same points of the video.

The cost is that the model with the shorter chunk loses some clean scores near the anomalies; the benefit is that the difference between the models measures the models and not the length of the chunk.

## The metrics

The metrics answer three different questions, one per level, and none of them replaces the others:

- at the **row** level: does the score rise where an anomaly happens and stay low elsewhere?
- at the **video** level: does the model find the anomaly in the video, and stay silent outside the anomalies?
- at the **event** level: how many anomalies are detected?

In total there are 15 metrics, plus their mean. The table below shows how each one is computed:

| Level | Metric | How it is computed |
| --- | --- | --- |
| row | AUC | on all judgeable rows, with average ranks: the probability that an event row ranks above a clean row |
| row | AP | rows by decreasing score; at every distinct score value all the rows that have it enter together: AP = Σ (R_k − R_{k−1}) · P_k |
| row | AUC_a macro, AP_a macro | one video at a time. For every video that contains anomalies and has both event rows and clean rows, the AUC (or the AP) is computed using only the rows of that video: it measures whether, within that video, the score is higher during the anomalies than in the rest. Then the mean over the videos is taken, and every video weighs the same |
| row | AUC_a micro, AP_a micro | the rows of all the videos that contain anomalies are put in a single list and a single AUC (or AP) is computed. Unlike the macro version, videos with more rows weigh more and rows of different videos are also compared; unlike the overall AUC, normal videos are left out |
| row | precision, recall, F1, FPR | every row is predicted positive if its score is ≥ θ |
| video | precision, recall, F1, FPR | every video gives up to two outcomes. If it has anomalies: TP if at least one of its event rows has a score ≥ θ, otherwise FN. If it has clean rows: FP if at least one of them has a score ≥ θ, otherwise TN. TP, FN, FP and TN are counted over all the videos, and then precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = their harmonic mean, FPR = FP/(FP+TN). The details are below |
| event | recall | anomalies detected / judgeable anomalies, at threshold θ. An anomaly longer than T is split into several event rows. The rows of each anomaly are grouped: the anomaly is detected if at least one of its judgeable rows has a score ≥ θ, otherwise it is missed. The judgeable anomalies are those with at least one judgeable row |

**The video level.** Every video can give up to two outcomes:

- an anomaly detection part, present only if the video has judgeable event rows:
    - it is a TP if at least one of these rows reaches the threshold θ,
    - otherwise a FN;
- a clean part, present if the video has judgeable clean rows:
    - it is a FP if at least one of these rows reaches θ,
    - otherwise a TN.

A normal video has only the clean part, while a video with anomalies has both: this way a false alarm outside the anomalies is counted even when the anomaly was detected, and that video gives a TP and a FP.

**The event level.** The event level is also computed from the row table. An anomaly longer than T is split into several event rows: the rows of each anomaly are grouped, and the anomaly is detected if at least one of its judgeable rows reaches θ. Since the rows of an anomaly cover it without gaps, this is equivalent to saying that at least one chunk that saw the anomaly has a score ≥ θ. An anomaly that no chunk overlaps has no judgeable rows, and is therefore not counted in the recall.

**The protocol mean.** To rank models with a single figure, the mean of six metrics is used, one for each question:

    mean = ( AUC + AUC_a macro + row F1 + (1 − row FPR) + video F1 + event recall ) / 6

The other nine metrics are reported next to the mean, but do not enter its computation.

## The threshold

The thresholded metrics (precision, recall, F1 and FPR of rows and videos, and the event recall) need a threshold θ: a row is considered positive if its score is greater than or equal to θ. Every model has a single threshold, computed once on the validation set and then used as it is on all test sets.

To fix it, the validation rows are built with the same rules as the test rows, and the Youden point is taken on the judgeable rows, that is, the threshold that maximizes TPR − FPR. The figure by which models are compared is the protocol mean, multiplied by 100.

If no validation set is available, the threshold can be passed by hand; in that case, however, the result is not the protocol's and must be declared as such.

## How results are reported and compared

Two numbers are comparable only if they are computed on the same rows: same data, same T, same guard.

For this reason every results table must declare:

- the data set;
- the chunk size of every model, the asking rate, T and the guard;
- the threshold θ of every model;
- the counts: judgeable rows, unjudgeable rows and anomalies.

**Models with different chunk sizes.** As explained in the section on the guard, in this case the guard is the longest chunk size in the table and holds for all models; if a model with a longer chunk is added, the others must be recomputed with the new guard.

## The code

This guide comes with the package `ovad_protocol`, which computes all the metrics described so far from the chunk scores and the labels. It requires Python 3.8 or later and numpy; the package's `README.md` file contains all the details. Its content is summarized below:

| File | What it is for |
| --- | --- |
| `evaluate_ovad.py` | the evaluation: one label file (triples) and one score file, plus the same for the validation set |
| `protocol_core/` | the computation of rows, threshold and metrics |
| `example/` | synthetic data in the right format and a script that evaluates them |
| `tests/test_protocol.py` | the tests |

Before using the package it is worth running the tests: `python tests/test_protocol.py` must end with `16 of 16 tests passed`.

**The labels.** One JSON file is provided, with one entry per video. Every entry contains the duration of the video and all its anomalies, as triples `[start, end, class]` with start and end in seconds; for a normal video the list is empty, but the key is still required. The class must be a non-empty string: the code checks it, but, as explained above, it does not enter the metrics:

```json
{"video_001": {"duration": 73.07, "anomalies": [[16.57, 24.03, "Fighting"], [40.1, 45.0, "Robbery"]]},
 "video_002": {"duration": 120.0, "anomalies": []}}
```

**The scores.** One JSON file is provided, with the same keys as the labels. It contains the model's hyperparameters and, for every video, the anomaly scores of the chunks: either as a list of numbers in grid order, in which case the code rebuilds the extents from chunk size and asking rate, or with the explicit extent of every chunk:

```json
{"hyperparameters": {"chunk_size": 8, "asking_rate": 4, "grid": "start"},
 "scores": {"video_001": [0.01, 0.02, 0.35, 0.40],
            "video_002": [{"start": 0.0, "end": 8.0, "score": 0.02}]}}
```

With the grid `"start"` chunk j covers the interval [j·h, j·h + c], clipped to the duration of the video. If the chunks were cut on frames it is better to pass the explicit extents: at 29.97 fps, for example, the chunk "from 4 to 12 s" actually goes from 4.004 to 12.012 s.

**How to run it.** With the threshold computed on the validation set:

```
python evaluate_ovad.py \
    --labels test/labels.json --scores test/scores.json \
    --val-labels val/labels.json --val-scores val/scores.json \
    --output results.json
```

As an alternative to the validation set, the threshold can be passed with `--threshold`. The `--guard` option is needed only when models with different chunk sizes are compared, and must be set to the longest chunk size among those compared; if it is not given, it equals the model's chunk size. The script prints a table with the 15 metrics, the protocol mean and the counts, and with `--output` saves all the results to JSON.
