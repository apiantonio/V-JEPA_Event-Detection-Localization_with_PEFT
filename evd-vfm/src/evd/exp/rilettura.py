"""Reread a closed run from the scores on disk, without GPU or model.

``scores.jsonl`` holds one score per window, so a closed run is re-analysed in
seconds. This module is the part shared by all re-analysers (strata in
``scripts/analisi_strati.py``, latency in ``scripts/ricalcola_latenza.py``): it opens
the folder, reads the scores, rebuilds the grid with the protocol sampler and pairs the
annotations. The geometry comes from the sampler, not the file (ADR-055), and the
partition must be stated, since in ``edp_v2`` validation has a contiguous grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from evd.annotations.schema import TemporalAnnotation
from evd.exp.demo import open_run
from evd.exp.sources import apply_overrides, build_dataset
from evd.infer.replay import replay_video, scores_from_disk
from evd.types import WindowScores
from evd.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class RereadRun:
    """A closed run, reopened from the scores on disk.

    :param run: object returned by :func:`~evd.exp.demo.open_run`, with threshold and
        config.
    :param bundle: the dataset, for fps, ``num_frames``, classes and annotations.
    :param scores: chunk scores per video on the grid of the requested partition.
    :param gt: annotations of the reread videos; a video missing from the annotation file
        becomes an empty :class:`~evd.annotations.schema.TemporalAnnotation`, and their
        count is logged.
    :param split: reread partition, which decided the generated grid.
    """

    run: Any
    bundle: Any
    scores: dict[str, list[WindowScores]]
    gt: dict[str, TemporalAnnotation]
    split: str

    @property
    def fps(self) -> dict[str, float]:
        """Return the native fps per video, as temporal metrics require."""
        return {v: self.bundle.meta(v).fps for v in self.scores}

    @property
    def num_frames(self) -> dict[str, int]:
        """Return the native frames per video."""
        return {v: self.bundle.meta(v).num_frames for v in self.scores}


def data_group(run: Any, overrides: list[str]) -> DictConfig:
    """Return the run ``data`` group, with path overrides applied.

    Paths recorded in a run are cluster paths (``/shared/...``), so local use needs
    overrides, which say where files are, never what is read.

    :param run: object returned by :func:`~evd.exp.demo.open_run`.
    :param overrides: ``key=value`` for the ``data`` group.
    :return: the resolved group.
    """
    group = OmegaConf.create(dict(run.cfg["data"]))
    apply_overrides(group, overrides)
    return group


def reread_run(run_dir: Path, *, split: str, overrides: list[str]) -> RereadRun:
    """Open a closed run and rebuild its scores and annotations.

    :param run_dir: the ``runs/{run_id}/`` folder.
    :param split: partition to reread (``test`` | ``val``); it decides which grid the
        sampler generates.
    :param overrides: ``key=value`` for the ``data`` group paths.
    :return: the reread run.
    :raises FileNotFoundError: if the run or its scores do not exist.
    :raises ValueError: if a grid window has no score in the file (usually the wrong
        partition).
    """
    run = open_run(run_dir)
    score_values = scores_from_disk(run.score_path, split=split)
    bundle = build_dataset(data_group(run, overrides))
    sampler = run.sampler(validazione=(split == "val"))

    scores = {
        video_id: replay_video(score_values, bundle.meta(video_id), sampler)
        for video_id in sorted(score_values)
    }
    raw = {v: bundle.annotations.get(v) for v in scores}
    without = sorted(v for v, a in raw.items() if a is None)
    if without:
        log.info(
            "%d of %d videos have no rows in the annotation file and count as normal (first: %s)",
            len(without),
            len(raw),
            without[0],
        )
    gt = {v: (a if a is not None else TemporalAnnotation(v)) for v, a in raw.items()}
    return RereadRun(run=run, bundle=bundle, scores=scores, gt=gt, split=split)
