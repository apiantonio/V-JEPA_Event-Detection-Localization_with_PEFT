# EVD Studio: the demonstrator on a PC

Double-click the demonstrator launcher (`run_demo.cmd` in the delivery package,
`Avvia dimostratore.cmd` in the development repository). The browser opens; keep the
launcher window open and press **Ctrl+C** in it to stop the service. No Internet
connection is needed.

## Watching a precomputed result

1. In **Model and seed**, choose the run. The default is **LoRA r32→VPT-shallow n8, seed 42**, the
   best mean of the UCF-Crime and MIVIA-Crime OVAD protocol means (`runs/classifiche/ovad_guard30`);
   without it DoRA r=32 seed 42, otherwise the first run listed. The ranking is chosen on the test
   sets, as those tables declare.
2. In **Saved results**, search for a video and select its card.
3. Press **Open the selected video**, then Play.

The library shows the local videos present in the run scores or manifest, filterable by
partition (Test, Validation, Train) and by Normal/Anomalous. A model evaluated on an
external test set adds that partition (**MIVIA-Crime test**), replayed from its external
evaluation (`runs/<run_id>/<run_id>__mivia/`) with the transferred threshold. **With
localization boxes** keeps only the videos with saved boxes (they carry a `boxes` tag):
the videos with spatial ground truth, 140 on UCF-Crime and 39 on MIVIA-Crime. Replaying
saved results does not load the model and does not use the GPU. Videos without scores need
a new inference and the model weights. The **Hide panel / Show panel** button toggles the
left column.

**List videos / List events** and **Sort by** order the library. Videos sort by the metrics of
that video (chunk AUC, AP, F1, precision, recall, accuracy, false-alarm rate, TP/FP/FN/TN, peak
score, duration, event F1/recall/precision, missed events, false alarms, detection latency, mean
AP or IoU of its events); events, one card per annotated interval, by the detector on that event
(detected, latency, share of its chunks above threshold, peak and mean score, duration) or by the
localization of the default method (event AP, event IoU). The button next to the menu reverses the
order; values that are not defined (AUC on a video without both classes, AP on an event the method
does not evaluate one by one) stay last, shown as `n/a`. The numbers are computed once per score
file, by the functions of the player page (`apps/demo/library.py`). Opening an event card starts
the video at that event.

Click the timeline to seek; click an alarm to open it. Speeds 0.5×–4× and full screen are
available. Arrow keys move by one second, space pauses, `[` and `]` change alarm, `0`
returns to the start. **Chunk outcome** (`V`), in the header of the timeline, colours its chunks by
verdict, as in the rendered movies: TP, FP, FN and TN, on the exact score against the threshold, so
the counts are those of the Metrics card. It needs an annotation, and hiding the annotations hides
it too.

## Analysing a file from the PC

1. Choose a model whose trained weights are available.
2. Open **New analysis** and select the file.
3. Check the device and the path of the V-JEPA weights, then press **Analyse the video**.

The file is copied to `runs/demo-local/uploads/`. A file selected here is always analysed
(a dataset video name is not enough to reuse its scores). **Cancel analysis** stops the
local process.

Saved runs work immediately in replay; to analyse new files with a model, copy its
`checkpoint/checkpoint.pt` into the run folder shown by the UI (checkpoints are not in
Git). The service looks for the backbone weights in `../data/checkpoints/`, with the name
recorded in the run config; the path can be corrected in the UI. The local batch is 1.
**CPU** works but is slow; **NVIDIA GPU** needs CUDA and enough memory for V-JEPA 2.1-L at
the native resolution (the demo never lowers the resolution or changes the protocol).
**Automatic** uses the GPU when available, otherwise the CPU.

## What the page shows

- **Anomaly score**: the model output; the threshold comes from the run validation.
- **Alarms**: intervals produced by the engine shared by evaluation and demonstrator.
- **Annotations**: the dataset reference, when available. Classes are not predicted by the
  model (binary output).
- **Waiting for the first window**: the causal score is not available yet.
- **Saved results / Local inference**: whether scores are re-read or computed now.

The timeline projects scores onto frames for display only; the metrics in the card
describe the run test set, not the single opened file.

## Localization boxes

When the selected model has saved localization boxes (`runs/<run_id>/localization/`, also
used by its MIVIA-Crime evaluation) and the video is annotated, the **Localization** card
lists every method evaluated for that model, grouped by family. Every video opens on the default
method of the detector: where the official table has been exported, its best rule (on LoRA
r32→VPT-shallow n8 the largest 4-connected component, under MultiClass-Branch→Concepts-AllChunks),
otherwise the best mean IoU on UCF-Crime. A choice in the menu lasts for the opened video.
The card shows the mean AP of the official methods, or the mean IoU of the others, on UCF-Crime and
MIVIA-Crime, the value of the annotated class and of the current event; **Metrics** has the
per-class table. Boxes are drawn on the video only inside
annotated intervals, in native pixels, with the same frame projection used by the
evaluation. Two independent buttons in the top bar show or hide each layer: **GT boxes**
(`G`, dashed ochre: the spatial ground truth of the frozen snapshot the IoU was computed
on, reviewed frames with a box of the event class) and **Method boxes** (`B`, solid and
blue). **Annotations** governs only the temporal annotation. Where a GT box exists the
card shows the **frame IoU**, the per-frame quantity the evaluation averages. Both layers
are hidden while Streaming. The default method is chosen on the test set, so the ranking
is exploratory.

**YOLO style** (`Y`, on by default) draws the method boxes with a thick line and no label; off,
they are thin and numbered.

**Interpolate planes** (`I`) moves each method box linearly on its corners between one
plane and the next, about 0.5 s apart, instead of jumping. It only changes the drawing: AP, IoU and
frame IoU stay on the evaluated boxes, and the card says `interpolated view`. Boxes pair by
decreasing IoU, from 0.1 up; a box without a partner keeps its place and appears or disappears at
the boundary, so nothing crosses the frame. It uses the following plane, so it is not causal.

To add the boxes of a model, export them once from the pulled campaign predictions:

```powershell
python scripts/export_demo_localization.py --run runs/<run_id> `
  --predictions <campaign>/pulled_annotated/<detector>/predictions_annotated `
  --report <campaign>/report_annotated<_detector> --campaign localization-auto-20261016
```

The export refuses the boxes if the chunk scores differ from the run's `scores.jsonl`. Add
`--ground-truth <campaign>/evaluation-only/gt-snapshot` (alone or with the other options) to
store the annotation boxes; it refuses a snapshot other than the evaluated one.

The class-routing campaign (`localization-class-routing-20261021`) is exported from its
evaluator results instead of the comparison tables, with the chunk scores of its IG passes and
a readable name per method; `--append` adds the methods to the export the run already has:

```powershell
python scripts/export_demo_localization.py --run runs/<run_id> `
  --predictions <campaign>/stages/<stage> --evaluation <run folder>/evaluation/<stage> `
  --passes <campaign>/passes/<detector>/test --campaign localization-class-routing-20261021 `
  --method "B2-A2-phase1=MultiClass-Full→Concepts-AllChunks, 55% threshold" --append
```

The page shows the readable name (the campaign id stays in the Metrics card). After a merge the
ranks are recomputed over all methods and the default follows the same rule (best UCF-Crime
mean).

The official table ranks box rules by offline AP, and its stages seal their material instead of
boxes. `--official-table` reads which rule a step of the table names best for the detector, builds
its boxes from the stage and makes it the default, which later merges keep:

```powershell
python scripts/export_demo_localization.py --run runs/<run_id> `
  --official-table <campaign>/evaluation/tabella_localizzazione.json --official-step ours-best `
  --official-stage <campaign>/stages/ign-n32-campaign `
  --official-name "Largest 4-conn. component, MultiClass-Branch→Concepts-AllChunks" `
  --intervals <campaign>/plan/real_use/intervals.json `
  --ground-truth <snapshot> --campaign localization-class-routing-20261021
```

AP has no operating point: it ranks every box by confidence. So the rule enters the menu four times:
θ@TPR90 (the default, the confidence at which 90% of the annotated frames of the true positive rows
keep a box, from the evaluation of the false alarms), θ@F1max (the confidence with the highest macro
F1@[0.3:0.05:0.7], offline at 1 fps; chosen on UCF-Crime and moved to MIVIA-Crime), the most
confident box of each plane, with no threshold, and all the boxes. The export recomputes the
per-event AP from the stage and refuses to proceed if it differs from the evaluation, or if the
chunk flags of the stage disagree with the run scores. The IoU of each entry is computed here, on
those very boxes; the AP is of the rule and does not change between them.

## Localization movies

`scripts/render_localization_videos.py` renders, for chosen methods, one movie per annotated
video in this page's layout (`apps/demo/localization_video.py`): a header with model,
localization method and video; the frame with the method boxes and the dashed ground truth;
a panel with the online score and alarm, the chunk the boxes come from with its TP/FP/FN/TN
verdict, the annotated events with their IoU and the frame IoU; and the timeline with the
chunks coloured by verdict. Steps `plan`, `render` (sharded, re-runnable), `catalog` (writes
`catalog.json` and `index.html` from `assets/localization_videos.html`) and `verify`. The
catalogue carries the evaluated IoU of every method on every event, and the page lists events
or videos sorted by it (best or mean of the shown methods, one method, the gap between two,
or the name); `node tests/localization_videos_ui.cjs [catalog.json]` checks the sorting.

## Manual start and other folders

From the project environment, inside `evd-vfm`:

```powershell
python apps/demo/serve.py
python apps/demo/serve.py --videos "D:\UCF videos" --videos "C:\Other videos"
python apps/demo/serve.py --runs "D:\results" --port 8765
```

Without `--videos`, `../data/raw/` is indexed; without `--port`, a free port is chosen.
The service listens on `127.0.0.1` only, publishes nothing on the Internet and uses no CDN.
The launcher looks for `%USERPROFILE%\miniconda3\envs\pytorch\python.exe`; set `EVD_PYTHON`
for a different installation (dependencies: `pyproject.toml`, including the `demo` extra).

## Produced files and common problems

Every opening is saved in `runs/demo-local/sessions/<id>/` (`demo.json`, HTML page,
`analysis.log`); uploads and sessions stay on disk and can be removed freely (they are not
official run results). `console.html` URLs are valid while the service runs.

For a portable package, with the video copied next to the page, use the CLI:

```powershell
python apps/demo/run_demo.py --run runs/e2d_ucf_vjepa21l_frozen-attentive_s42 --video "C:\path\Arson016_x264.mp4" --modo replay
```

The CLI also writes the annotated video and the PNG timeline (`--no-video` skips the MP4).

If a video does not play, the player shows a message: **MP4 with H.264** is the most
compatible choice (the `.mp4` extension alone does not identify the codec). If the library
is empty, check the video folder and the presence of `config.yaml`, `metrics.json` and
`scores.jsonl` in the run. An out-of-memory error in a new analysis requires the CPU or a
larger GPU.

## Hop and window

In **New analysis** choose **Hop (seconds)** and **Window (seconds)** (8 s on EDP-v2)
before starting; `hop <= window`, both positive. Windows keep 32 frames, so a shorter
window increases the temporal density of the samples. A different duration changes the
context with respect to training and the threshold is not recalibrated. The new hop is
saved in `demo.json` together with the original one and the `demo_override` marker;
protocol and official scores are unchanged, and replay always keeps the saved grid.

From the CLI: `--modo live --hop-s 2`, or `--modo live --window-s 4 --hop-s 2`. With
`--modo auto`, an explicit hop or window forces a new inference.

## Live analysis

1. Choose a library video or a file from the PC.
2. Select **Live analysis (1×)**, the hop and the device.
3. Press **Prepare live**: the video appears; wait for the model to load.
4. Press **Start video** when the model is ready. The video plays at 1×, the timeline grows
   with the scores just computed, and a red notice flags a window above threshold.

**Pause** stops new frames; **Cancel live** stops the process and keeps the partial
results; at the end **Open full result** opens the usual player. The source is a file
played as a timed stream: the engine waits for the last frame of each window before
decoding and scoring it (no precomputed scores). **Analysed up to**, **Distance from the
video** and **Last window compute** show how far the model lags behind playback. Each
session keeps `progress.jsonl`, `clock.json` and, when complete, `demo.json` and the HTML
page (`realtime: true`, `clock: browser_playback_1x`).

## Per-video metrics

Replay, local analysis and live mode show the global model summary and the metrics of the
current video: window precision, recall, F1, AUC/AP and accuracy, TP/FP/FN/TN, normal
windows with a false alarm, and event metrics (provisional during live mode). The grid
latency (median and p90) uses `detection_latency` with the `intersezione` rule: annotated
onset -> end of the first window above threshold intersecting the event; it excludes
compute time. Without annotations only score counts and compute times are shown.
