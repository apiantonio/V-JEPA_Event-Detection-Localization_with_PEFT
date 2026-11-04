"""A run becomes a demonstrator: opening, score source, result.

The demonstrator draws, this module
decides where the numbers come from. Two sources:

* live: build the backbone, apply the adaptation, load the selected weights from the
  checkpoint and go through :class:`InferenceEngine`, exactly the evaluation path
  (invariant 4); works on any video, including unseen ones;
* replay: scores are read back from ``scores.jsonl`` without building a
  model; only for videos the run evaluated, to show the thesis numbers without a GPU.

Both modes use the run geometry by default; an explicit hop requires new inference and
is demo-only. Windows always come from
:func:`~evd.protocol.chunks.sampler_for_protocol` and the projection from
:func:`~evd.infer.engine.video_result`.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from evd.data.decoder import make_decoder
from evd.eval.demo_metrics import compute_timing
from evd.eval.frame_map import FrameScoreMapper
from evd.eval.scorers import HeadScorer
from evd.infer.engine import InferenceEngine, VideoResult, video_result
from evd.infer.replay import replay_video, scores_from_disk
from evd.protocol.chunks import sampler_for_protocol
from evd.types import EventClass, PreprocessSpec, VideoMeta
from evd.utils.io import read_json
from evd.utils.logging import get_logger

if TYPE_CHECKING:  # no torch or annotations at import time for callers that only open a run
    from evd.annotations.schema import TemporalAnnotation
    from evd.data.windows import CausalWindowSampler

log = get_logger(__name__)

#: Ways the demonstrator obtains scores. ``auto`` picks ``replay`` when the video is
#: already in the run scores (the numbers reported in the thesis).
MODES = ("auto", "live", "replay")

#: Name of the score file inside a run folder (ADR-051).
SCORE_NAME = "scores.jsonl"


@dataclass(frozen=True)
class RunDemo:
    """A run opened for the demonstrator: config, identity, operating point.

    :param dir: the ``runs/{run_id}/`` folder.
    :param run_id: run identity, from the folder name.
    :param cfg: resolved config the run used, as a dict.
    :param metrics: content of ``metrics.json``, or ``{}`` if the run did not write it.
    :param threshold: operating threshold of the run.
    :param threshold_source: how it was estimated (``val_youden`` on the active protocol).
    """

    dir: Path
    run_id: str
    cfg: Mapping[str, Any]
    metrics: Mapping[str, Any]
    threshold: float
    threshold_source: str

    @property
    def protocol(self) -> Mapping[str, Any]:
        return self.cfg["protocol"]

    @property
    def score_path(self) -> Path:
        return self.dir / SCORE_NAME

    @property
    def checkpoint_path(self) -> Path:
        return self.dir / "checkpoint" / "checkpoint.pt"

    def sampler(
        self,
        spec: PreprocessSpec | None = None,
        *,
        validazione: bool = False,
        hop_s: float | None = None,
        window_s: float | None = None,
    ) -> CausalWindowSampler:
        """Return the run grid for the requested partition.

        ``spec`` is ``None`` in replay, where no backbone exists: ``num_frames`` and
        ``target_fps`` are then read from the protocol (checked against the backbone by
        ``ExperimentRunner._check_geometry`` at every run).

        :param spec: backbone preprocessing, or ``None``.
        :param validazione: whether the validation grid is needed (contiguous in ``edp_v2``,
            ADR-047).
        :return: the sampler.
        """
        protocol = demo_geometry(self.protocol, hop_s=hop_s, window_s=window_s)
        return sampler_for_protocol(spec, protocol, validazione=validazione)


def validate_hop(value: float) -> float:
    """Validate the hop requested for a new local inference.

    :raises ValueError: if the hop is not finite and positive.
    """
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("The hop must be a finite number greater than zero.")
    return value


def demo_geometry(protocol, *, hop_s=None, window_s=None):
    """Apply local geometry overrides on a copy; the historical geometry is not modified.

    :raises ValueError: on an invalid window, a non-fixed grid or a hop larger than the
        window.
    """
    if hop_s is None and window_s is None:
        return protocol
    temporal = dict(protocol["temporal"])
    if hop_s is not None:
        temporal["hop_s"] = temporal["val_hop_s"] = validate_hop(hop_s)
    if window_s is not None:
        window_s = float(window_s)
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("The window must be finite and greater than zero.")
        if temporal.get("window_grid", "endpoint") != "fixed":
            raise ValueError("A modifiable duration requires the fixed grid.")
        temporal["window_s"] = window_s
    if temporal.get("window_s") is not None and temporal["hop_s"] > temporal["window_s"]:
        raise ValueError("The hop must not exceed the window duration.")
    return {**protocol, "temporal": temporal}


def open_run(run_dir: str | Path) -> RunDemo:
    """Open a run folder and extract what is needed to demonstrate it.

    :param run_dir: the ``runs/{run_id}/`` folder.
    :return: the run, with its operating point.
    :raises FileNotFoundError: if the folder or its ``config.yaml`` does not exist.
    :raises ValueError: if the run declares no threshold (it did not reach evaluation).
    """
    d = Path(run_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a run folder")
    config = d / "config.yaml"
    if not config.exists():
        raise FileNotFoundError(
            f"{config} does not exist: the folder is not a run, or the run died before "
            f"writing the config"
        )
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config} does not contain a mapping")
    metrics: Mapping[str, Any] = {}
    metrics_path = d / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
    threshold = metrics.get("threshold")
    if not isinstance(threshold, Mapping) or "value" not in threshold:
        raise ValueError(
            f"{d.name} declares no threshold in metrics.json: the run did not reach "
            f"evaluation. The operating point is protocol (docs/04 §5.3) and is not chosen here"
        )
    return RunDemo(
        dir=d,
        run_id=d.name,
        cfg=cfg,
        metrics=metrics,
        threshold=float(threshold["value"]),
        threshold_source=str(threshold.get("source", "ignota")),
    )


def run_classes(run: RunDemo) -> tuple[EventClass, ...]:
    """Return the class catalogue of the run dataset, to name an event.

    Classes are not stored in the run config: the canonical UCF-Crime catalogue is used and
    its size checked against the config. Used only to name annotated intervals (the head is
    binary, ADR-031 point 4).

    :return: the 13 anomaly classes plus ``Normal``, in index order.
    """
    from evd.annotations.ucf_classes import ucf_classes

    catalog = ucf_classes()
    expected = int(run.cfg.get("data", {}).get("n_classes", len(catalog)))
    if expected != len(catalog):
        log.warning(
            "the run declares %d classes and the UCF-Crime catalogue has %d: event "
            "names may not match",
            expected,
            len(catalog),
        )
    return catalog


def annotation(
    directory: str | Path | Sequence[str | Path] | None, meta: VideoMeta
) -> TemporalAnnotation | None:
    """Return the ground truth of a video, if annotations are reachable.

    Used only to draw the comparison lane; no model decision sees it. Several folders can
    be given (UCF-Crime and MIVIA-Crime deliveries): each is tried and the first containing
    the video wins; :func:`~evd.annotations.ucf_lab.delivery_of` recognizes the delivery
    from its ``.txt`` name.

    :param directory: a delivery folder, a sequence of folders, or ``None``.
    :param meta: the video, whose ``num_frames`` is used to clamp intervals.
    :return: the annotation, or ``None`` if no folder contains the video.
    :raises ValueError: if an annotation file is malformed or cites a class outside the
        catalogue (from :meth:`UcfLabAnnotations.load`).
    """
    if directory is None:
        return None
    from evd.annotations.ucf_lab import UcfLabAnnotations, delivery_of

    folders = [directory] if isinstance(directory, (str, Path)) else list(directory)
    for folder in folders:
        base = Path(folder)
        if not base.is_dir():
            continue
        try:
            delivery = delivery_of(base)
        except FileNotFoundError:
            # A folder without a known delivery is not an error: it may belong to another dataset.
            log.debug("%s does not contain a known annotation delivery", base)
            continue
        source = UcfLabAnnotations.from_dir(
            base, delivery=delivery, num_frames={meta.video_id: meta.num_frames}
        )
        found = source.load().get(meta.video_id)
        if found is not None:
            return found
    return None


def analyze(
    run: RunDemo,
    video: str | Path,
    *,
    modo: str = "auto",
    device: str | None = None,
    off_threshold: float | None = None,
    min_len: int = 1,
    batch_size: int | None = None,
    backbone_checkpoint: str | Path | None = None,
    hop_s: float | None = None,
    window_s: float | None = None,
    timing: dict | None = None,
) -> tuple[VideoResult, str]:
    """Return the demonstrator result on a video: with the model, or with the scores on disk.

    :param run: run opened by :func:`open_run`.
    :param video: the video file.
    :param modo: one of :data:`MODES`; ``auto`` picks ``replay`` if the video is in the run
        scores, ``live`` otherwise.
    :param device: forward device in ``live`` mode; ``None`` uses the run one, then the
        environment.
    :param off_threshold: hysteresis closing threshold; ``None`` means the run threshold.
    :param min_len: minimum interval length, in frames.
    :param batch_size: windows per batch in ``live`` mode; ``None`` uses the run ``eval``
        value.
    :param backbone_checkpoint: backbone weights location, if not where the run had them
        (see :func:`live_engine`).
    :return: the result and the mode actually used (to report next to the numbers).
    :raises ValueError: on an unknown mode, or ``replay`` on a video the run did not
        evaluate.
    :raises FileNotFoundError: if what the chosen mode needs is missing (scores for
        ``replay``, checkpoint for ``live``).
    """
    if modo not in MODES:
        raise ValueError(f"unknown mode {modo!r}: allowed {list(MODES)}")
    demo_geometry(run.protocol, hop_s=hop_s, window_s=window_s)
    if hop_s is not None or window_s is not None:
        if modo == "replay":
            raise ValueError("Changing hop or window requires a new inference: use live mode.")
    decoder = make_decoder(str(run.cfg.get("data", {}).get("decoder", "pyav")))
    meta = decoder.probe(video)
    mapper = FrameScoreMapper(policy=str(run.protocol["temporal"]["frame_mapping"]))

    chosen = modo
    if modo == "auto":
        chosen = (
            "replay"
            if hop_s is None and window_s is None and _replay_covers(run, meta.video_id)
            else "live"
        )
        log.info("mode %s chosen for %s", chosen, meta.video_id)

    if chosen == "replay":
        score_values = scores_from_disk(run.score_path)
        # The grid follows the video partition: in `edp_v2` validation has a contiguous grid
        # (ADR-047), and the test sampler would find none of its windows.
        known = score_values.get(meta.video_id)
        windows = replay_video(
            score_values,
            meta,
            run.sampler(validazione=bool(known and known.validazione)),
        )
        return (
            video_result(
                mapper,
                meta,
                windows,
                threshold=run.threshold,
                off_threshold=off_threshold,
                min_len=min_len,
            ),
            "replay",
        )

    engine, _ = live_engine(
        run,
        device=device,
        batch_size=batch_size,
        backbone_checkpoint=backbone_checkpoint,
        hop_s=hop_s,
        window_s=window_s,
    )
    started = time.perf_counter()
    if timing is not None and engine.batch_size == 1:
        costs, scores, tick = [], [], [started]

        def before_window(window):
            tick[0] = time.perf_counter()

        for item in engine.stream(video, meta, before_window=before_window):
            costs.append((time.perf_counter() - tick[0]) * 1000)
            scores.append(item)
        timing.update(compute_timing(costs))
        result = video_result(
            engine.mapper,
            meta,
            scores,
            threshold=run.threshold,
            off_threshold=off_threshold,
            min_len=min_len,
        )
    else:
        result = engine.run_video(
            video, meta, threshold=run.threshold, off_threshold=off_threshold, min_len=min_len
        )
    if timing is not None:
        timing["analysis_s"] = time.perf_counter() - started
        timing["mean_window_ms"] = 1000 * timing["analysis_s"] / max(1, len(result.window_scores))
    return result, "live"


def _replay_covers(run: RunDemo, video_id: str) -> bool:
    """:return: whether the run scores on disk cover this video."""
    if not run.score_path.exists():
        return False
    return video_id in scores_from_disk(run.score_path)


#: Hydra unresolved mandatory value; in ``head.n_classes`` it means "decided by the
#: runner" and survives in the run ``config.yaml``.
REQUIRED_PLACEHOLDER = "???"


def n_classes_from_checkpoint(head_weights: Mapping[str, Any], embed_dim: int) -> int:
    """:return: how many logits the head saved in these weights emits.
    :raises ValueError: if no recognizable classifier is among the weights.

    The classifier is found by shape ``[n_classes, embed_dim]``, as the last 2-D tensor with
    that second dimension (pooler ``Linear`` layers come first).
    """
    candidati = [
        int(tensor.shape[0])
        for name, tensor in head_weights.items()
        if name.endswith(".weight")
        and getattr(tensor, "ndim", 0) == 2
        and int(tensor.shape[1]) == embed_dim
    ]
    if not candidati:
        raise ValueError(
            f"no [n, {embed_dim}] classifier among the head weights: "
            f"available keys {sorted(head_weights)[:8]}"
        )
    return candidati[-1]


def resolve_n_classes(
    head_cfg: Mapping[str, Any], head_weights: Mapping[str, Any], embed_dim: int
) -> int:
    """:return: the run head logits, from the config or the weights.
    :raises ValueError: if the config declares a number contradicted by the weights.

    ``head.n_classes`` may be ``???`` (decided by :meth:`ExperimentRunner._n_classes`) and
    the resolved value is stored only in the checkpoint; a declared number is verified
    against the weights.
    """
    declared = head_cfg.get("n_classes", REQUIRED_PLACEHOLDER)
    from_weights = n_classes_from_checkpoint(head_weights, embed_dim)
    if str(declared).strip() == REQUIRED_PLACEHOLDER or declared is None:
        return from_weights
    if int(declared) != from_weights:
        raise ValueError(
            f"head.n_classes declares {declared} but the checkpoint weights have "
            f"{from_weights}: config and checkpoint do not belong to the same run"
        )
    return from_weights


def live_engine(
    run: RunDemo,
    *,
    device: str | None = None,
    batch_size: int | None = None,
    backbone_checkpoint: str | Path | None = None,
    hop_s: float | None = None,
    window_s: float | None = None,
    num_workers: int = 0,
    provenienza: dict[str, Any] | None = None,
) -> tuple[InferenceEngine, PreprocessSpec]:
    """Rebuild the run model and put it inside the inference engine.

    Follows ``ExperimentRunner.run``: backbone from the registry, pre-adaptation if declared
    (``preadapt.prepare_backbone``), head from the registry, adaptation from the factory,
    then the selected weights (those of the step optimizing the selection criterion, e.g.
    step 4600 rather than 5524 on E2d). No feature cache: the demonstrator measures the
    real pipeline.

    :param run: the open run.
    :param device: forward device; ``None`` uses the run one, then the environment.
    :param batch_size: windows per batch; ``None`` uses the ``eval`` value.
    :param num_workers: decoding workers for ``run_videos`` only; 0 keeps the one-video-at-a
        -time demonstrator behaviour. Evaluation has no augmentation, so it changes no pixel.
    :param backbone_checkpoint: backbone weights location, if not where the run had them
        (paths recorded on the cluster do not exist on a laptop); content is still verified
        by ``checkpoint_sha`` and ``strict=True`` loading.
    :param provenienza: if given, the ``fondi`` merge receipt is written into it as
        ``preadattamento``, for runs derived from this one (``exp/esterno.py``).
    :return: the engine and the backbone ``PreprocessSpec``.
    :raises FileNotFoundError: if the run checkpoint or the backbone weights are missing,
        naming which one.
    :raises ValueError: if the checkpoint belongs to another run.
    """
    import torch

    from evd.exp.preadapt import prepare_backbone
    from evd.exp.runner import ExperimentRunner, _arguments
    from evd.models.adapt.factory import AdaptContext, build_adaptation
    from evd.models.backbones.base import BACKBONES
    from evd.models.heads.base import HEADS
    from evd.train.checkpoint import restore_adapter_state
    from evd.utils import env

    demo_geometry(run.protocol, hop_s=hop_s, window_s=window_s)
    cfg = run.cfg
    # Both preconditions are checked before building the backbone (minutes and GBs on
    # ViT-L).
    if not run.checkpoint_path.exists():
        raise FileNotFoundError(
            f"{run.checkpoint_path} does not exist: without weights the 'live' mode has nothing "
            f"to run. Copy the run `checkpoint/` folder from the cluster, or use "
            f"`--modo replay` on a video the run evaluated (`--elenca` lists them)"
        )
    backbone_arguments = _arguments(cfg["backbone"])
    if backbone_checkpoint is not None:
        backbone_arguments["checkpoint"] = str(backbone_checkpoint)
    # Only for backbones with weights: `TinyBackbone` is built from a seed and declares no
    # `checkpoint`.
    declared = backbone_arguments.get("checkpoint")
    if declared and not Path(str(declared)).exists():
        raise FileNotFoundError(
            f"the backbone weights are not in {declared}: that is the path of the machine the "
            f"run ran on. The file is the same: pass "
            f"`--backbone-checkpoint <local path>`"
        )

    device_name = str(device or cfg.get("device") or env.device())
    backbone = BACKBONES.get(str(cfg["backbone"]["id"]))(**backbone_arguments, device=device_name)
    spec = backbone.preprocess_spec()
    n_blocks = len(backbone.blocks())

    # The checkpoint is read before building the head: it is the only artifact knowing the
    # head logits when the config does not declare them (see `n_classes_from_checkpoint`).
    data = torch.load(run.checkpoint_path, map_location="cpu", weights_only=True)
    identity = data.get("identity") or {}
    found = str(identity.get("run_id", run.run_id))
    if found != run.run_id:
        raise ValueError(
            f"the checkpoint in {run.checkpoint_path} belongs to run {found!r}, not "
            f"{run.run_id!r}: the folder was copied from the wrong place"
        )
    # Selected weights if present, last-step weights otherwise (a run interrupted before the
    # first evaluation has no selection); the log says which.
    head_weights = data.get("selected_head") or data["head"]
    adapter_weights = data.get("selected_adapter") or data.get("adapter") or {}

    n_classes = resolve_n_classes(cfg["head"], head_weights, backbone.embed_dim)
    # `n_classes` overrides the config field as in `ExperimentRunner._build`.
    head = HEADS.get(str(cfg["head"]["id"]))(
        embed_dim=backbone.embed_dim,
        **{**_arguments(cfg["head"]), "n_classes": n_classes},
    )
    context = AdaptContext(n_classes, n_blocks, int(cfg.get("seed", 0)))
    # `fondi` pre-adaptation before `apply()`, as in the runner (ADR-065): without it the
    # prompts would run on the bare backbone and still load (same keys), giving plausible
    # wrong numbers (`docs/09` R24).
    _, receipt = prepare_backbone(cfg, backbone, context=context, device=device_name)
    if provenienza is not None and receipt is not None:
        provenienza["preadattamento"] = receipt
    strategy = build_adaptation(cfg["adapt"], context)
    strategy.apply(backbone)

    log.info(
        "weights from the checkpoint: last step %s, selection at step %s, %d logits",
        data.get("step"),
        data.get("selected_step", "missing"),
        n_classes,
    )
    head.load_state_dict({k: v.to(device_name) for k, v in head_weights.items()})
    head.to(device_name).eval()
    restore_adapter_state(strategy, adapter_weights)

    levels = ExperimentRunner._levels(
        cfg,
        n_blocks=n_blocks,
        hierarchical=tuple(backbone.hierarchical_layers()),
        store_layers=None,
    )
    evaluation = cfg.get("eval", {})
    scorer = HeadScorer(
        backbone,
        head,
        levels=levels,
        normal_idx=int(cfg["protocol"].get("classes", {}).get("normal_idx", 0)),
        device=device_name,
        precision=str(evaluation.get("precision", "fp32")),
    )
    engine = InferenceEngine(
        scorer,
        run.sampler(spec, hop_s=hop_s, window_s=window_s),
        FrameScoreMapper(policy=str(run.protocol["temporal"]["frame_mapping"])),
        make_decoder(str(cfg.get("data", {}).get("decoder", "pyav"))),
        spec,
        batch_size=int(batch_size or evaluation.get("batch_size", 4)),
        num_workers=int(num_workers),
    )
    return engine, spec
