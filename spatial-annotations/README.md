# Reviewed spatial annotations

Spatial annotations prepared and reviewed with Forma for the test events used in the
localization study. These are evaluation annotations, not bounding-box training labels.
Only frames explicitly marked `reviewed` are included; drafts and excluded frames are omitted.

| Quantity | FS-UCF-Crime test | MIVIA-Crime test |
|---|---:|---:|
| Included videos | 140 | 39 |
| Completed events | 276 | 44 |
| Reviewed native frames | 131,428 | 9,051 |
| Reviewed frames with no box | 542 | 10 |
| Native-frame box instances | 351,511 | 22,400 |
| Contiguous review segments | 9,002 | 678 |

UCF covers all 140 anomalous test videos. MIVIA covers a **subset**, 39 of the 135
anomalous test videos; the rest are not spatially annotated by this release. Normal
videos are not covered. Annotations may originate from model proposals corrected or
accepted during manual review: `origin` preserves this provenance. They are not a
blind, independent annotation study; proposal-assisted review can bias evaluation.

## Files and coordinates

- `ucf_crime/<video_id>.json` and `mivia_crime/<video_id>.json`: one record per video.
- `manifest.json`: source snapshot hashes, verified counts and SHA-256 of every video record.
- `videos.csv`: coverage and box counts per video.
- `export_reviewed.py`: export and round-trip verification, using only the Python standard library.

Frame indices are **zero-based**. Frame intervals `[start, end]` and native-pixel boxes
`[x0, y0, x1, y1]` are **inclusive**. For COCO-style `xywh`, use
`[x0, y0, x1-x0+1, y1-y0+1]`. `width`, `height`, `fps`, `num_frames` and the source-video
SHA-256 identify the native video geometry. `events` contains the temporal class labels
of the included video; `reviewed_segments` contains the spatial annotation.

A segment holds identical boxes on every frame from `start` through `end`. A reviewed
segment with `boxes: []` is explicitly reviewed and empty. A frame absent from all
segments is **unannotated**, not a confirmed negative. Review segments are not tracks:
there is no persistent object identity across segments. The stored review unit documents
the temporal holding of an annotation; it is not an independent annotation on each frame.

```python
import json

video = json.load(open("ucf_crime/Shoplifting049_x264.json", encoding="utf-8"))
frame = 1200
boxes = next((s["boxes"] for s in video["reviewed_segments"]
              if s["start"] <= frame <= s["end"]), None)
# None: unannotated; []: reviewed empty; otherwise native-pixel rectangles.
```

## Download and provenance

Browse the JSON files directly or download the two dataset archives from the
[annotations-v1 release](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/releases/tag/annotations-v1).
The same annotation release is mirrored in the private development repository.

The export uses one consistent read-only SQLite backup. It expands each written JSON
record and checks every reviewed frame, class and rectangle against that snapshot.
The working database, history, reviewer names, free-text notes, machine paths, caches,
draft proposals and video/image files are not distributed.

To regenerate from your own Forma project:

```bash
python spatial-annotations/export_reviewed.py --project /path/to/forma-project \
  --output spatial-annotations --archives /path/to/release-archives
```

Annotation contributions follow the repository's [LICENSE](../LICENSE). Source videos
and third-party dataset material retain their own terms; this release grants no rights
to those videos. Cite the thesis and the respective UCF-Crime / FS-UCF-Crime or MIVIA-Crime
dataset source when using the annotations.
