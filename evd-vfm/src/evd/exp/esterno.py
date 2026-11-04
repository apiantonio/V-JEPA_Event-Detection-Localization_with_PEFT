"""A closed run, evaluated on a dataset it has never seen.

How well does a model trained on UCF-Crime hold on another surveillance dataset.
No training, hyper-parameter choice or
threshold estimation: an inference pass with the selected weights of a matrix cell,
followed by the docs/04 §4.3 metrics. Reused components:

* :func:`~evd.exp.demo.live_engine` rebuilds the model from the run ``config.yaml`` and
  loads the selected weights;
* :class:`~evd.infer.engine.InferenceEngine` does the windowing (invariant 4);
* :func:`~evd.eval.valutazione.metriche_di_valutazione` builds the same table as
  ``ExperimentRunner._evaluate``;
* :class:`~evd.exp.tracker.LocalTracker` writes the folder and registry row.

The threshold is not re-estimated: MIVIA-Crime has no validation set, so the source run
threshold is transferred and recorded as ``fissa``. AUC/AP measure ranking on the new
domain, while P/R/F1 also depend on calibration; ``soglia_oracolo`` reports the same
metrics at the Youden point computed on MIVIA-Crime itself, an upper bound that is not
protocol.

Governed by docs/03 §6.1 (single engine) and docs/04 §4.3 (metrics).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evd.annotations.schema import TemporalAnnotation
from evd.eval import metrics as M
from evd.eval.valutazione import metriche_di_valutazione
from evd.exp.demo import live_engine, open_run
from evd.exp.sources import DatasetBundle, build_dataset, videos_of
from evd.exp.tracker import LocalTracker
from evd.types import WindowScores
from evd.utils.io import read_jsonl, write_jsonl
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["ExternalOutcome", "validation_negative_scores", "evaluate_external_dataset"]


@dataclass(frozen=True)
class ExternalOutcome:
    """What the cross-dataset evaluation produced.

    :param run_id: identity of the new run, ``{source}__{dataset}``.
    :param run_sorgente: run providing weights and threshold.
    :param dataset: dataset measured on.
    :param run_dir: written folder, a full run (``config.yaml``, ``metrics.json``,
        ``scores.jsonl``) replayable by the demonstrator.
    :param metrics: the metrics, as in ``metrics.json``.
    """

    run_id: str
    run_sorgente: str
    dataset: str
    run_dir: Path
    metrics: dict[str, Any]


def validation_negative_scores(percorso: str | Path) -> list[float]:
    """Return the scores of the validation negative windows of a closed run.

    Used by the threshold sensitivity analysis, which must keep the source run negatives.

    :param percorso: ``runs/{run_id}/scores.jsonl`` of the source run.
    :return: the scores, or an empty list if the file is missing or has no validation rows.
    """
    p = Path(percorso)
    if not p.exists():
        return []
    return [
        float(r["score"])
        for r in read_jsonl(p)
        if str(r.get("split", "test")) == "val" and int(r.get("label", -1)) == 0
    ]


def _oracle(
    chunk: Mapping[str, Sequence[WindowScores]],
    gt: Mapping[str, TemporalAnnotation],
    tau_source: str,
) -> dict[str, Any]:
    """Return thresholded metrics at the Youden point computed on this dataset.

    Not protocol and not deployable: the threshold is read from the chunks it judges, so it
    is optimistic by construction. It separates ranking loss from calibration shift.

    :param chunk: chunk scores per video.
    :param gt: annotations of the evaluated videos.
    :param tau_source: provenance of the protocol threshold, recorded in the ``confronto``
        field.
    :return: the oracle threshold and the two dependent triples.
    """
    rows = M.chunk_records(chunk, gt)
    value, _ = M.threshold_youden(
        [float(r["score"]) for r in rows], [int(r["label"]) for r in rows]
    )
    return {
        "nota": (
            "SECONDARY ANALYSIS, not protocol: tau estimated on the same chunks it "
            "judges, hence optimistic by construction. It only separates ranking loss "
            "from calibration loss with respect to the 'chunk_prf' row, "
            f"whose threshold has provenance '{tau_source}'."
        ),
        "threshold": value,
        "chunk_prf": M.dataset_chunk_prf(chunk, gt, value, source="fissa").as_dict(),
        "video_prf": M.video_decision_prf(chunk, gt, value, source="fissa").as_dict(),
    }


def _check_compatibility(bundle: DatasetBundle, cfg: Mapping[str, Any]) -> int:
    """Check that the new dataset and the run share the same classes.

    Per-class metrics skip ``Normal`` by index: diverging catalogues would shift every row
    by one (see :func:`~evd.exp.runner.protocol_normal_idx`).

    :param bundle: dataset being evaluated.
    :param cfg: config of the source run.
    :return: the agreed index of ``Normal``.
    :raises ValueError: if protocol and catalogue disagree.
    """
    from evd.exp.runner import protocol_normal_idx

    idx = protocol_normal_idx(cfg["protocol"], bundle)
    expected = int((cfg.get("data") or {}).get("n_classes", bundle.n_classes()))
    if expected != bundle.n_classes():
        raise ValueError(
            f"the run was trained on a dataset with {expected} classes and "
            f"{bundle.name} has {bundle.n_classes()}: per-class tables would not be "
            f"comparable"
        )
    return idx


def evaluate_external_dataset(
    run_dir: str | Path,
    data_cfg: Mapping[str, Any],
    *,
    device: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    backbone_checkpoint: str | Path | None = None,
    video_ids: Sequence[str] | None = None,
    official: bool = True,
    tracker: LocalTracker | None = None,
) -> ExternalOutcome:
    """Evaluate the selected weights of a closed run on an external dataset.

    The output folder is ``runs/{run_id}/{run_id}__{dataset_id}``, inside the source run
    folder.

    :param run_dir: ``runs/{run_id}/`` of the source run; it needs ``checkpoint/`` and a
        threshold in ``metrics.json``.
    :param data_cfg: resolved ``data`` group of the external dataset, with paths valid on
        this machine.
    :param device: forward device; ``None`` uses the run one, then the environment.
    :param batch_size: windows per batch; ``None`` uses the run ``eval`` value.
    :param num_workers: decoding workers; ``None`` uses the run ``eval`` value (affects only
        time).
    :param backbone_checkpoint: backbone weights, if not where the run had them.
    :param video_ids: subset of videos for a trial; ``None`` means all. A subset does not
        produce thesis numbers and ``metrics.json`` declares ``sottoinsieme: true``.
    :param ufficiale: whether the run enters the registry as official (requires a clean
        working tree, invariant 2).
    :param tracker: for tests; ``None`` uses the environment runs folder.
    :return: the outcome, with the written folder.
    :raises FileNotFoundError: if checkpoint or backbone weights are missing.
    :raises ValueError: if the run declares no threshold or the class catalogues disagree.
    """
    run = open_run(run_dir)
    bundle = build_dataset(data_cfg)
    normal_idx = _check_compatibility(bundle, run.cfg)
    evaluation = run.cfg.get("eval") or {}

    chosen = tuple(video_ids) if video_ids is not None else bundle.query_videos
    if not chosen:
        raise ValueError(
            f"{bundle.name} has no evaluation videos: `query_videos` is empty, "
            f"so there is nothing to measure"
        )
    log.info(
        "external evaluation: run %s (tau %.6f, %s) on %s, %d videos",
        run.run_id,
        run.threshold,
        run.threshold_source,
        bundle.name,
        len(chosen),
    )

    # Pre-adaptation receipt of a `fondi` cell, copied into the new `meta.json`: the only
    # written proof of which backbone was measured (`docs/09` R24).
    reconstruction: dict[str, Any] = {}
    engine, spec = live_engine(
        run,
        device=device,
        batch_size=batch_size,
        backbone_checkpoint=backbone_checkpoint,
        num_workers=int(
            num_workers if num_workers is not None else evaluation.get("num_workers", 0)
        ),
        provenienza=reconstruction,
    )
    results = engine.run_videos(videos_of(bundle, chosen))
    chunk = {v: r.window_scores for v, r in results.items()}
    # A video without annotation rows and a video without events lower recall in the same way
    # but are different things: the MIVIA-Crime datasource raises on the former, so `or` only
    # covers datasets that do not.
    gt = {v: (bundle.annotations.get(v) or TemporalAnnotation(video_id=v)) for v in chosen}

    val_negatives = validation_negative_scores(run.score_path)
    threshold = run.threshold
    # `fissa`, not `val_youden`: the threshold was estimated on the validation of another
    # dataset (ADR-057).
    source = "fissa"
    metrics = metriche_di_valutazione(
        chunk=chunk,
        gt=gt,
        soglia=threshold,
        sorgente=source,
        classi=bundle.classes,
        normal_idx=normal_idx,
        fps={v: bundle.meta(v).fps for v in chosen},
        num_frames={v: bundle.meta(v).num_frames for v in chosen},
        protocol=run.cfg["protocol"],
        requests=evaluation.get("metrics") or {},
        negative_scores=val_negatives,
    )
    metrics.update(
        {
            "eval_split": "test",
            "eval_precision": str(evaluation.get("precision", "fp32")),
            "n_eval_videos": len(chosen),
            "n_chunk_eval": sum(len(v) for v in chunk.values()),
            "n_test_videos": len(chosen),
            "n_chunk_test": sum(len(v) for v in chunk.values()),
            "threshold": {
                "value": threshold,
                "source": source,
                # Where the threshold actually comes from, which `fissa` alone does not say.
                "trasferita_da": run.run_id,
                "regola_originale": run.threshold_source,
                "n_scores": 0,
                "n_negativi": len(val_negatives),
                "scores_from": "run_sorgente",
            },
            "soglia_oracolo": _oracle(chunk, gt, run.threshold_source),
            "valutazione_esterna": {
                "run_sorgente": run.run_id,
                "dataset": bundle.name,
                "annotation_source": bundle.annotation_source,
                "sottoinsieme": video_ids is not None,
                # The geometry is the run's, the condition for putting this table next to the UCF-Crime
                # one.
                "geometria": {
                    k: run.protocol["temporal"][k]
                    for k in ("window_grid", "window_s", "hop_s", "num_frames", "target_fps")
                    if k in run.protocol["temporal"]
                },
                "preprocess_hash": spec.hash(),
            },
        }
    )

    tracker = tracker or LocalTracker()
    run_id = f"{run.run_id}__{bundle.dataset_id}"
    # The new run config is the source one with only the `data` group replaced.
    new_cfg = {
        **dict(run.cfg),
        "data": dict(data_cfg),
        "valutazione_esterna": {"run_sorgente": run.run_id, "dataset": bundle.name},
    }
    handle = tracker.start(
        run_id,
        new_cfg,
        # The folder lives inside the source run folder; the `run_id` does not change
        # (`open_run` takes it from the folder name).
        dentro=run.run_id,
        official=official,
        regime={"eval_precision": metrics["eval_precision"], "training": False},
        protocol_sha=bundle.annotation_source,
        extra_meta={
            "valutazione_esterna": metrics["valutazione_esterna"],
            "threshold": metrics["threshold"],
            **reconstruction,
        },
        # ADR-057: nothing from the environment governs here (regime, batch and precision come
        # from the source `config.yaml`, no seed is used); the variables are recorded as
        # provenance under the key saying they did not govern.
        ignored_env=("EVD_SEED", "EVD_PRECISION", "EVD_BATCH_SIZE", "EVD_TF32"),
    )
    # Scores on disk in the matrix-run format (ADR-051), so the folder is replayable and
    # readable by `eval/report.py`. The partition is `test`; no validation row is invented.
    path = write_jsonl(
        handle.dir / "scores.jsonl",
        [{**r, "split": "test"} for r in M.chunk_records(chunk, gt)],
    )
    log.info("scores of %d windows written to %s", metrics["n_chunk_eval"], path)
    tracker.finish(
        handle,
        metrics,
        extra={"exp": "esterno", "dataset": bundle.name, "run_sorgente": run.run_id},
    )
    return ExternalOutcome(
        run_id=run_id,
        run_sorgente=run.run_id,
        dataset=bundle.name,
        run_dir=handle.dir,
        metrics=metrics,
    )
