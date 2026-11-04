"""Input geometry measured on the real domain.

On surveillance video two mechanisms can behave differently:
small, distant events may benefit from the finer 384 tokenization, and the official
centre view discards 25% of the width, which may contain the event.

This module only builds the split; the factorial design is :mod:`evd.exp.e0e`, already
dataset-independent.

Why the pilot could be redone (19 August)
-----------------------------------------

E0f v1 stopped at calibration: UCF-Crime training videos had only video-level labels,
so most windows sampled from an ``Arson`` video contained no fire, and the expected
effect was below the metric resolution. The FS-UCF-Crime annotations provide temporal
intervals on training videos too, so every window can carry the label of its content.

The task
--------

14-way classification per window:

1. Positive for class c: the window ends inside an annotated ``c`` interval (last-frame
   rule, docs/04 §4.1); the class comes from the interval, never the folder.
2. Normal: the whole window span is disjoint from every interval, with a guard margin.
3. Normal windows come from the same anomaly videos, removing the scene shortcut.

``window_id`` does not contain resolution or crop variant, so with fixed
``num_frames``, ``target_fps`` and hop every cell sees the same windows with the same
labels; only the spatial geometry changes.

The split is built from ``Anomaly_Train.txt`` only: :func:`build_split` checks the
intersection with ``Anomaly_Test.txt`` and raises if it is not empty (the official
action-recognition splits overlap the test set on 171 of 700 videos).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from evd.annotations.schema import NORMAL_CLASS_NAME, TemporalAnnotation, window_class_label
from evd.annotations.ucf_classes import normal_index, ucf_classes
from evd.annotations.ucf_lab import UcfLabAnnotations
from evd.data.inventory import InventoryRecord, index_by_video_id, read_records
from evd.data.windows import CausalWindowSampler
from evd.types import VideoMeta, WindowSpec
from evd.utils.io import write_json
from evd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class E0fSplitConfig:
    """How the real-domain pilot is built, with per-window labels.

    :param anomaly_train: official ``Anomaly_Detection_splits/Anomaly_Train.txt``.
    :param anomaly_test: ``Anomaly_Test.txt``, read only to check the empty intersection.
    :param inventory_csv: UCF-Crime inventory (fps, frames, ``status``).
    :param annotations_dir: ``data/annotations/fs_ucf_crime/``.
    :param destinazione: folder where the split files are written.
    :param num_frames: frames per window; a plan constant, not an axis.
    :param target_fps: requested temporal rate; with ``num_frames`` and hop it fixes the
        timestamp grid and thus the window identities.
    :param hop_s: step between consecutive windows.
    :param finestre_per_video: windows of a class a video must offer to be admitted to
        that class (the same number ``subsample_support`` requires).
    :param video_per_classe_train: training videos per class.
    :param video_per_classe_val: disjoint validation videos per class.
    :param margine_normale_s: guard band between a normal window and the nearest event.
    :param seed: sampling seed, so the split can be rebuilt.
    """

    anomaly_train: Path
    anomaly_test: Path
    inventory_csv: Path
    annotations_dir: Path
    destinazione: Path
    num_frames: int = 16
    target_fps: float = 7.5
    hop_s: float = 1.0
    finestre_per_video: int = 4
    video_per_classe_train: int = 16
    video_per_classe_val: int = 8
    margine_normale_s: float = 1.0
    seed: int = 1

    def __post_init__(self) -> None:
        if self.finestre_per_video <= 0:
            raise ValueError("finestre_per_video must be positive")
        if self.video_per_classe_train <= 0 or self.video_per_classe_val <= 0:
            raise ValueError("both training and validation videos are required")
        if self.hop_s <= 0 or self.target_fps <= 0 or self.num_frames <= 0:
            raise ValueError("hop_s, target_fps and num_frames must be positive")


def _video_id_from_row(row: str) -> str | None:
    """Turn ``Abuse/Abuse001_x264.mp4`` (or with backslashes) into ``Abuse001_x264``."""
    value = row.strip().replace("\\", "/")
    if not value or value.startswith("#"):
        return None
    return PurePosixPath(value).stem


def read_official_split(percorso: Path) -> set[str]:
    """Read an official UCF-Crime split and return its ``video_id`` values.

    :param percorso: split file, one line per video.
    :return: identifiers without extension.
    :raises FileNotFoundError: if the file does not exist.
    """
    if not percorso.is_file():
        raise FileNotFoundError(f"official split does not exist: {percorso}")
    ids = set()
    for row in percorso.read_text(encoding="utf-8-sig").splitlines():
        video_id = _video_id_from_row(row)
        if video_id is not None:
            ids.add(video_id)
    return ids


def label_windows(
    ann: TemporalAnnotation,
    finestre: list[WindowSpec],
    *,
    nomi: dict[int, str],
    normal_idx: int,
    frame_margin: int,
) -> list[tuple[WindowSpec, str]]:
    """Label every window of a video, or leave it out.

    * ends inside an interval: the class of that interval;
    * its span is farther than the margin from every interval: ``Normal``;
    * in between (ends outside the event but still contains part of it): no label, left
      out of the task.

    :param ann: video annotation.
    :param finestre: complete windows of the video, in temporal order.
    :param nomi: class index to name.
    :param normal_idx: index of the ``Normal`` class.
    :param frame_margin: guard band, in native frames.
    :return: ``(window, class name)`` pairs in temporal order, for labelable windows only
        (the order is part of the contract).
    """
    outside: list[tuple[WindowSpec, str]] = []
    for window in finestre:
        class_name = window_class_label(ann, window.end_frame, normal_idx)
        if class_name != normal_idx:
            outside.append((window, nomi[class_name]))
            continue
        start, end = window.frame_span
        far = all(e + frame_margin < start or s - frame_margin > end for s, e, _ in ann.intervals)
        if far:
            outside.append((window, NORMAL_CLASS_NAME))
    return outside


def _windows_of(record: InventoryRecord, config: E0fSplitConfig) -> list[WindowSpec]:
    meta: VideoMeta = record.to_meta()
    sampler = CausalWindowSampler(
        num_frames=config.num_frames,
        target_fps=config.target_fps,
        hop_s=config.hop_s,
    )
    return sampler.windows_for_video(meta)


def _equispaced(finestre: list[str], how_many: int) -> list[str]:
    """Return ``how_many`` windows evenly spaced in an already time-ordered list."""
    if how_many >= len(finestre):
        return list(finestre)
    if how_many <= 0:
        return []
    if how_many == 1:
        return finestre[:1]
    step = (len(finestre) - 1) / (how_many - 1)
    return [finestre[round(i * step)] for i in range(how_many)]


def balance_per_class(
    per_class: dict[str, dict[str, list[str]]], how_many: int
) -> dict[str, list[str]]:
    """Reduce each class to ``how_many`` windows, spread over its videos.

    Balances the evaluation side: otherwise ``Normal`` would dominate and a head always
    answering "normal" would exceed 95%. The quota is spread as evenly as possible over the
    videos, the remainder going to the richest ones in deterministic order.

    :param per_classe: class -> video -> windows, time-ordered.
    :param how_many: windows to keep per class.
    :return: class -> chosen windows.
    """
    choices: dict[str, list[str]] = {}
    for class_name, per_video in per_class.items():
        video = sorted(per_video, key=lambda v: (-len(per_video[v]), v))
        base, rest = divmod(how_many, len(video))
        taken: list[str] = []
        for i, video_id in enumerate(video):
            quota = base + (1 if i < rest else 0)
            taken.extend(_equispaced(per_video[video_id], quota))
        # Videos without enough windows for their quota are compensated from the richer ones,
        # instead of returning a smaller class.
        if len(taken) < how_many:
            remaining = [w for v in video for w in per_video[v] if w not in set(taken)]
            taken.extend(remaining[: how_many - len(taken)])
        choices[class_name] = sorted(taken)
    return choices


def _write(percorso: Path, righe: list[str]) -> None:
    percorso.write_text("\n".join(righe) + "\n", encoding="utf-8", newline="\n")


def build_split(config: E0fSplitConfig) -> dict[str, Any]:
    """Write the pilot split and the per-window label map.

    Classes are served from the poorest (a video carrying intervals of two classes can be
    assigned to only one); an assigned video still contributes labels to every class in its
    intervals.

    :param config: split configuration.
    :return: the summary, also written to disk as ``split.json``.
    :raises ValueError: if the split touches ``Anomaly_Test`` or a class lacks enough
        eligible videos for the requested budget.
    """
    allowed_test = read_official_split(config.anomaly_test)
    id_train = read_official_split(config.anomaly_train)
    records = index_by_video_id(read_records(config.inventory_csv))
    catalog = ucf_classes()
    names = {c.idx: c.name for c in catalog}
    normal = normal_index(catalog)
    annotations = UcfLabAnnotations.from_dir(
        config.annotations_dir,
        num_frames={k: r.num_frames for k, r in records.items()},
    ).load()

    # --- 1. labels of the windows of every eligible video ----------------------
    labels_per_video: dict[str, list[tuple[WindowSpec, str]]] = {}
    counts: dict[str, Counter[str]] = {}
    discarded = {"non_in_inventario": 0, "status": 0, "senza_annotazione": 0, "troppo_corti": 0}
    for video_id in sorted(id_train):
        record = records.get(video_id)
        if record is None:
            discarded["non_in_inventario"] += 1
            continue
        if record.status != "ok":
            discarded["status"] += 1
            continue
        ann = annotations.get(video_id)
        if ann is None or not ann.intervals:
            # Videos without events are not needed: the Normal class comes from the quiet parts of
            # anomaly videos.
            discarded["senza_annotazione"] += 1
            continue
        windows = _windows_of(record, config)
        if not windows:
            discarded["troppo_corti"] += 1
            continue
        map_ = label_windows(
            ann,
            windows,
            nomi=names,
            normal_idx=normal,
            frame_margin=max(1, round(config.margine_normale_s * record.fps)),
        )
        labels_per_video[video_id] = map_
        counts[video_id] = Counter(class_name for _, class_name in map_)

    # --- 2. which videos can serve which class ---------------------------------
    requested = config.video_per_classe_train + config.video_per_classe_val
    candidati: dict[str, list[str]] = defaultdict(list)
    for video_id, count in counts.items():
        for class_label, how_many in count.items():
            if class_label == NORMAL_CLASS_NAME:
                # `Normal` is not assigned to any video: it comes from the quiet parts of all chosen
                # videos.
                continue
            # Same threshold `subsample_support` will require, so calibration does not fail midway.
            if how_many >= config.finestre_per_video:
                candidati[class_label].append(video_id)

    poor = {c: len(v) for c, v in candidati.items() if len(v) < requested}
    if poor:
        raise ValueError(
            f"budget cannot be met: {poor} have fewer than {requested} videos with "
            f"at least {config.finestre_per_video} windows of their class. "
            "Lower video_per_classe_train/val or finestre_per_video instead of "
            "loosening the threshold: a video admitted with fewer windows breaks "
            "the parity across cells the budget exists to guarantee"
        )

    # --- 3. assignment: the poorest class chooses first ------------------------
    rng = random.Random(config.seed)
    assigned: set[str] = set()
    train: list[tuple[str, str]] = []
    val: list[tuple[str, str]] = []
    chosen_per_class: dict[str, list[str]] = {}
    for class_label in sorted(candidati, key=lambda c: (len(candidati[c]), c)):
        free_items = [v for v in sorted(candidati[class_label]) if v not in assigned]
        if len(free_items) < requested:
            raise ValueError(
                f"{class_label}: {len(free_items)} videos still free, {requested} needed. "
                "Classes are served from the poorest to the richest precisely to "
                "avoid this: if it still happens, the per-class cap is too high"
            )
        # Sort by label richness, then sample: at equal extraction cost a richer video brings
        # more examples.
        sorted_items = sorted(free_items, key=lambda v: (-counts[v][class_label], v))[
            : 2 * requested
        ]
        chosen = sorted(rng.sample(sorted_items, requested))
        assigned.update(chosen)
        chosen_per_class[class_label] = chosen
        train.extend((class_label, v) for v in chosen[: config.video_per_classe_train])
        val.extend((class_label, v) for v in chosen[config.video_per_classe_train :])

    everyone = {v for _, v in train} | {v for _, v in val}
    contaminated = everyone & allowed_test
    if contaminated:
        raise ValueError(
            f"{len(contaminated)} pilot videos appear in Anomaly_Test "
            f"({sorted(contaminated)[:5]}): choosing the geometry on them would make "
            "every number reported on that test set selected on test data"
        )

    # --- 4. label map, restricted to the chosen videos -------------------------
    # Labels of other classes carried by an assigned video are true and kept, but a video
    # enters a class only if it offers at least `finestre_per_video` windows of it.
    pilot_train_ids = {v for _, v in train}
    collection: dict[str, dict[str, dict[str, list[str]]]] = {"train": {}, "val": {}}
    for video_id in sorted(everyone):
        side = "train" if video_id in pilot_train_ids else "val"
        per_class_video: dict[str, list[str]] = defaultdict(list)
        for window, class_label in labels_per_video[video_id]:
            per_class_video[class_label].append(window.window_id)
        for class_label, class_windows in per_class_video.items():
            if len(class_windows) >= config.finestre_per_video:
                collection[side].setdefault(class_label, {})[video_id] = class_windows

    train_normals = len(collection["train"].get(NORMAL_CLASS_NAME, {}))
    if train_normals < config.video_per_classe_train:
        raise ValueError(
            f"only {train_normals} videos offer at least {config.finestre_per_video} "
            f"normal windows, {config.video_per_classe_train} needed. The Normal "
            "class is not assigned: it comes from the quiet parts of anomaly videos, "
            "so either events cover almost everything or the guard margin is too wide"
        )

    classes = sorted(set(collection["train"]) & set(collection["val"]))
    only_one_side = (set(collection["train"]) | set(collection["val"])) - set(classes)
    if only_one_side:
        raise ValueError(
            f"classes present on one side only: {sorted(only_one_side)}. A class "
            "without evaluation windows cannot be measured, and one without "
            "training windows cannot be learned: either way the 14-way top-1 "
            "would describe a different task"
        )

    # The evaluation side is balanced: the poorest class sets the quota.
    available_val = {c: sum(len(w) for w in collection["val"][c].values()) for c in classes}
    val_quota = min(available_val.values())
    val_choices = balance_per_class({c: collection["val"][c] for c in classes}, val_quota)

    labels: dict[str, str] = {}
    per_side: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    for class_label in classes:
        for class_windows in collection["train"][class_label].values():
            for window_id in class_windows:
                labels[window_id] = class_label
                per_side["train"][class_label] += 1
        for window_id in val_choices[class_label]:
            labels[window_id] = class_label
            per_side["val"][class_label] += 1
    config.destinazione.mkdir(parents=True, exist_ok=True)
    _write(config.destinazione / "pilota-train.txt", [f"{c}/{v}.mp4" for c, v in sorted(train)])
    _write(config.destinazione / "pilota-val.txt", [f"{c}/{v}.mp4" for c, v in sorted(val)])
    _write(config.destinazione / "pilota-all-videos.txt", sorted(everyone))
    _write(
        config.destinazione / "pilota-classInd.txt",
        [f"{i} {c}" for i, c in enumerate(classes, start=1)],
    )
    _write(config.destinazione / "pilota-classes.txt", classes)
    write_json(
        config.destinazione / "pilota-window-labels.json",
        {
            "regola": "last frame (docs/04 §4.1): the window carries the class "
            "of the interval covering its end_frame; Normal only if the whole "
            "span is farther from events than the declared margin",
            "target_fps": config.target_fps,
            "num_frames": config.num_frames,
            "hop_s": config.hop_s,
            "margine_normale_s": config.margine_normale_s,
            "classi": classes,
            "etichette": labels,
        },
    )

    summary = {
        "sorgente_split": str(config.anomaly_train),
        "sorgente_annotazioni": str(config.annotations_dir),
        "n_classi": len(classes),
        "classi": classes,
        "n_video": len(everyone),
        "n_video_train": len(train),
        "n_video_val": len(val),
        "n_finestre_etichettate": len(labels),
        "finestre_train_per_classe": dict(sorted(per_side["train"].items())),
        "finestre_val_per_classe": dict(sorted(per_side["val"].items())),
        "quota_val_per_classe": val_quota,
        "val_disponibili_per_classe": dict(sorted(available_val.items())),
        "classe_val_piu_povera": min(available_val, key=lambda c: available_val[c]),
        "video_per_classe": {
            "train": config.video_per_classe_train,
            "val": config.video_per_classe_val,
        },
        "finestre_per_video": config.finestre_per_video,
        "candidati_per_classe": {c: len(v) for c, v in sorted(candidati.items())},
        "scartati": discarded,
        "geometria": {
            "num_frames": config.num_frames,
            "target_fps": config.target_fps,
            "hop_s": config.hop_s,
        },
        "seed": config.seed,
        "intersezione_con_anomaly_test": 0,
        "nota": (
            "Anomaly_Test was read only to check that the intersection is empty: "
            "the geometry is not chosen on the test set"
        ),
    }
    write_json(config.destinazione / "split.json", summary)
    log.info(
        "E0f: %d classes, %d videos (%d train, %d val), %d labelled windows",
        len(classes),
        len(everyone),
        len(train),
        len(val),
        len(labels),
    )
    return summary
