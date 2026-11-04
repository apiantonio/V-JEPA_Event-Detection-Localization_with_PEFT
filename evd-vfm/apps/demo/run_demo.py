#!/usr/bin/env python
r"""The demonstrator: from (video, run) to the artifacts. A thin CLI (ADR-015).

    python apps/demo/run_demo.py --video path/to/video.mp4 \
        --run runs/e2d_ucf_vjepa21l_frozen-attentive_s42

    # without a checkpoint, on videos the run already evaluated (no GPU)
    python apps/demo/run_demo.py --video ... --run ... --modo replay

    # what can be replayed from this run
    python apps/demo/run_demo.py --run ... --elenca

Writes to ``--out`` (default ``runs/{run_id}/demo/{video_id}/``):

* ``index.html``: the interactive console, opened with a double click;
* ``annotated.mp4``: the video with the decision overlaid, for presentations;
* ``timeline.png``: the figure, for slides and the thesis;
* ``demo.json``: the payload, to redo the other three without running anything.

No decision is taken here: mode, threshold and geometry come from :mod:`evd.exp.demo`,
drawing from the renderers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Lets the script run without `pip install -e .`, like `run_experiment.py`.
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from apps.demo import attention as attention_map  # noqa: E402
from apps.demo import localization, overlay, payload, timeline, viewer  # noqa: E402
from evd.eval.demo_metrics import metriche_video, saved_labels  # noqa: E402
from evd.exp.demo import MODES, analyze, annotation, open_run, run_classes  # noqa: E402
from evd.exp.demo_stream import analyze_direct  # noqa: E402
from evd.exp.explain import analyze_attention, write_artifacts  # noqa: E402
from evd.infer.replay import available_videos, scores_from_disk  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("run_demo")

#: Run metrics shown by the page and the video: the three the thesis uses to describe a
#: cell (`docs/04` §4.3).
SHOWN_METRICS = ("chunk_auc", "chunk_ap", "chunk_auc_anomalous")

#: Geometry fields describing a window, for the caption.
GEOMETRY_FIELDS = ("window_s", "hop_s", "num_frames", "target_fps", "window_grid", "frame_mapping")

#: Annotation deliveries on this machine, tried in order when `--annotazioni` is not
#: given; the file that names the video wins.
LOCAL_ANNOTATIONS = (
    _ROOT.parent / "data/annotations/fs_ucf_crime",
    _ROOT.parent / "data/mivia_crime",
)


def main(argv: list[str] | None = None) -> int:
    """Run the demonstrator.

    :param argv: the arguments, ``None`` for ``sys.argv``.
    :return: the process exit code.
    :raises ValueError: if the live stream is requested with incompatible options.
    """
    args = _parser().parse_args(argv)
    setup_logging()
    run = open_run(args.run)

    if args.elenca:
        # The partition is printed because it changes the grid: on `edp_v2` validation has
        # contiguous windows (ADR-047).
        known = scores_from_disk(run.score_path)
        print(f"{run.run_id}: {len(known)} videos replayable in 'replay' mode")
        for v in available_videos(known):
            print(f"  {known[v].split:<5s} {v}")
        return 0

    if not args.video:
        print("--video is required (or --elenca to list what the run covers)", file=sys.stderr)
        return 2

    video = Path(args.video)
    if not video.exists():
        print(f"{video} does not exist", file=sys.stderr)
        return 2

    timing = {}
    options = {
        "device": args.device,
        "backbone_checkpoint": args.backbone_checkpoint,
        "hop_s": args.hop_s,
        "window_s": args.window_s,
        "timing": timing,
        "off_threshold": args.off_threshold,
        "min_len": args.min_len,
    }
    if args.realtime:
        if args.modo == "replay" or not args.out or not args.playback_clock:
            raise ValueError("The live stream requires live mode, --out and --playback-clock.")
        result = analyze_direct(
            run,
            video,
            events=Path(args.out) / "progress.jsonl",
            clock=Path(args.playback_clock),
            annotations=args.annotazioni,
            **options,
        )
        mode = "live"
    else:
        result, mode = analyze(
            run,
            video,
            modo=args.modo,
            batch_size=args.batch_size,
            **options,
        )
    log.info(
        "%s: %d windows, %d intervals above tau=%.4f (mode %s)",
        result.meta.video_id,
        len(result.window_scores),
        len(result.intervals),
        run.threshold,
        mode,
    )

    out = Path(args.out) if args.out else run.dir / "demo" / result.meta.video_id
    out.mkdir(parents=True, exist_ok=True)

    # The video next to the artifacts: the folder becomes a package that can be zipped and
    # opened on another machine.
    source = _carries_video(video, out, copy=not args.no_copia_video)

    data = payload_for(
        run,
        result,
        mode,
        video_src=source,
        annotations=args.annotazioni,
        timing=timing,
        hop_s=args.hop_s,
        window_s=args.window_s,
        realtime=args.realtime,
    )

    written = [data.save_json(out / "demo.json"), viewer.scrivi(data, out / "index.html")]
    if not args.no_timeline:
        written.append(timeline.scrivi(data, out / "timeline.png"))
    if not args.no_video:
        log.info("rendering annotated.mp4 over %d frames…", data.num_frames)
        written.append(overlay.scrivi(data, video, out / "annotated.mp4", fourcc=args.fourcc))
    if args.attention:
        written.extend(_attention_maps(run, video, out, mode, args))

    print(f"\ndemonstrator ready in {out}")
    for p in written:
        print(f"  {p.name}")
    print(f"\nopen: {out / 'index.html'}")
    return 0


def _attention_maps(run, video: Path, out: Path, modo: str, args) -> list[Path]:
    """Add the two attention-map mp4 files to the demonstrator folder (ADR-063).

    Live mode only, with an explicit refusal: attention weights are an internal state of
    the forward pass, not a number saved in ``scores.jsonl``.

    :param run: the open run.
    :param video: the video file.
    :param out: the demonstrator artifact folder.
    :param modo: where the scores come from (``live`` or ``replay``).
    :param args: the CLI arguments.
    :return: the written files.
    :raises ValueError: if requested in ``replay`` mode.
    :raises RuntimeError: if the ADR-063 fidelity check fails.
    """
    if modo != "live":
        raise ValueError(
            "--attention requires 'live' mode: in replay scores are read from disk and "
            "no model runs, so no attention map exists. Pass "
            "`--modo live` (the run checkpoint and the backbone weights are required)"
        )
    log.info("attention maps: recomputing the pooler weights…")
    result = analyze_attention(
        run,
        video,
        device=args.device,
        backbone_checkpoint=args.backbone_checkpoint,
        batch_size=args.batch_size or 1,
        hop_s=args.hop_s,
        window_s=args.window_s,
    )
    if not result.fedele:
        raise RuntimeError(
            f"the attention recomputation deviates by {result.scarto_fedelta_max:.3g} "
            f"from the pooler output ({result.dtype_forward}): the map is NOT what the "
            f"model computed and is not shown"
        )
    written = write_artifacts(result, out)
    lo, hi = result.contrasto.estremi_in_uniforme()
    for cut, name in (("overlay", "attention.mp4"), ("affiancato", "attention_affiancato.mp4")):
        written.append(
            attention_map.scrivi(
                video,
                out / name,
                heat=result.heat,
                anomaly=result.anomaly_per_frame,
                threshold=result.threshold,
                fps=result.fps,
                video_id=result.video_id,
                run_id=result.run_id,
                layout=cut,
                fourcc=args.fourcc,
                divergente=result.contrasto.divergente,
                scale_lo=lo,
                scale_hi=hi,
                contrast_mode=result.contrasto.modo,
            )
        )
    return written


def payload_for(
    run,
    result,
    mode: str,
    *,
    video_src: str,
    annotations=None,
    timing: dict | None = None,
    hop_s: float | None = None,
    window_s: float | None = None,
    realtime: bool = False,
    export: Path | None = None,
) -> payload.DemoPayload:
    """Assemble what the page, the timeline and the videos show about an analysed video.

    The single author of the payload: the demonstrator and the localization movies
    (``scripts/render_localization_videos.py``) both pass from here.

    :param run: the open run.
    :param result: the analysis of :func:`evd.exp.demo.analyze`.
    :param mode: the mode actually used (``live`` or ``replay``).
    :param video_src: how the page reaches the video, relative to it.
    :param annotations: annotation delivery folders; ``None`` tries
        :data:`LOCAL_ANNOTATIONS`.
    :param timing: measured analysis times, for the video metrics.
    :param hop_s: demonstrator hop override, if any.
    :param window_s: demonstrator window override, if any.
    :param realtime: whether the page follows the browser clock.
    :param export: run folder holding the localization export; ``None`` looks next to the
        run (:func:`apps.demo.localization.export_dir`).
    :return: the payload.
    """
    # Several annotation folders exist (UCF-Crime and MIVIA-Crime deliveries); `annotazione`
    # tries them in order and returns `None` if none names the video (the "unseen video" case
    # of PdD §6).
    annotations_dir = annotations or [d for d in LOCAL_ANNOTATIONS if d.is_dir()]
    truth = annotation(annotations_dir, result.meta)
    video_metrics = metriche_video(
        result.window_scores,
        result.meta,
        run.threshold,
        annotation=truth,
        timing=timing,
        threshold_source=run.threshold_source,
        labels=saved_labels(run.score_path, result.meta.video_id) if mode == "replay" else None,
    )
    return payload.build_catalog(
        result,
        run_id=run.run_id,
        modo=mode,
        threshold=run.threshold,
        threshold_source=run.threshold_source,
        video_src=video_src,
        annotation=truth,
        metriche_video=video_metrics,
        classi=run_classes(run),
        metriche={k: run.metrics[k] for k in SHOWN_METRICS if isinstance(run.metrics.get(k), float)}
        | _video_f1(run)
        | {"eval_split": run.metrics.get("eval_split", "test")},
        protocollo=_effective_geometry(run, result.meta.video_id, mode, hop_s, window_s)
        | ({"realtime": True, "clock": "browser_playback_1x"} if realtime else {}),
        localization=_localization(run, result.meta, truth, export),
    )


def _localization(run, meta, truth, export: Path | None = None) -> dict | None:
    """Return the run's saved localization boxes for the video, if any.

    Boxes are drawn only inside annotated intervals, so a video without annotation (an
    uploaded file) gets none. An external evaluation finds them in its source run. A
    geometry mismatch is logged and the boxes are dropped: the analysis itself stays valid.

    :param run: the open run.
    :param meta: the opened video.
    :param truth: its annotation, or ``None``.
    :param export: run folder holding the export, if not next to the run.
    :return: the payload block, or ``None``.
    """
    base = export or localization.export_dir(run.dir, run.metrics)
    if truth is None or base is None:
        return None
    try:
        block = localization.for_video(
            base,
            meta.video_id,
            num_frames=meta.num_frames,
            width=meta.width,
            height=meta.height,
            intervals=[(int(s), int(e)) for s, e, _ in truth.intervals],
        )
    except ValueError as exc:
        log.warning("localization boxes not shown: %s", exc)
        return None
    if block is not None:
        log.info("localization: %d methods, default %s", len(block["methods"]), block["default"])
    return block


def _video_f1(run) -> dict[str, float]:
    """:return: ``{'video_f1': ...}`` if the run measured video-level P/R/F1."""
    prf = run.metrics.get("video_prf")
    if isinstance(prf, dict) and isinstance(prf.get("f1"), float):
        return {"video_f1": float(prf["f1"])}
    return {}


def _effective_geometry(
    run, video_id: str, modo: str, hop_s: float | None = None, window_s: float | None = None
) -> dict:
    """:return: the displayed geometry, with the validation hop when actually used."""
    temporal = run.protocol["temporal"]
    geometry = {k: temporal[k] for k in GEOMETRY_FIELDS if k in temporal}
    if modo == "replay":
        saved = scores_from_disk(run.score_path)[video_id]
        if saved.validazione:
            geometry["hop_s"] = temporal.get("val_hop_s", temporal["hop_s"])
    if modo == "live" and hop_s is not None:
        geometry["original_hop_s"] = temporal["hop_s"]
        geometry["hop_s"] = hop_s
        geometry["demo_override"] = True
    if modo == "live" and window_s is not None:
        geometry["original_window_s"] = temporal.get("window_s")
        geometry["window_s"] = window_s
        geometry["demo_override"] = True
    return geometry


def _carries_video(video: Path, out: Path, *, copy: bool) -> str:
    """Put the video where the page finds it, and return the relative path.

    With ``copia`` false a relative path from the artifact folder to the original file is
    written (saves space, ties the folder to this disk); if no relative path exists
    (another volume on Windows) the file is copied anyway.

    :param video: the original file.
    :param out: the artifact folder.
    :param copia: whether to copy the file next to the page.
    :return: the value for ``<video src=…>``.
    """
    if not copy:
        # `os.path.relpath` rather than `Path.relative_to(walk_up=True)`, which needs Python 3.12
        # (the project runs on 3.10, invariant 12).
        try:
            return Path(os.path.relpath(video.resolve(), out.resolve())).as_posix()
        except ValueError:
            log.warning("%s is on a different volume from %s: copying it anyway", video, out)
    destination = out / video.name
    if not destination.exists() or destination.stat().st_size != video.stat().st_size:
        shutil.copy2(video, destination)
    return video.name


def _parser() -> argparse.ArgumentParser:
    """:return: the CLI parser."""
    p = argparse.ArgumentParser(
        description="Demonstrator: video → score timeline → detected intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--window-s",
        type=float,
        help="local window duration in seconds (default: run, 8 s on EDP-v2)",
    )
    p.add_argument("--realtime", action="store_true", help="timed progressive inference")
    p.add_argument("--playback-clock", help="playback clock, provided by the console")
    p.add_argument(
        "--run", required=True, help="the runs/{run_id}/ folder to take the weights from"
    )
    p.add_argument("--video", help="the video to analyse")
    p.add_argument("--out", help="output folder (default: runs/{run_id}/demo/{video_id})")
    p.add_argument("--modo", choices=MODES, default="auto", help="where the scores come from")
    p.add_argument("--hop-s", type=float, help="time step in seconds between local inferences")
    p.add_argument("--device", default=None, help="where the forward pass runs in live mode")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="windows per forward; 1 reduces local memory",
    )
    p.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="backbone weights, if not where the run had them: the config records the path "
        "of the machine it ran on, which does not exist elsewhere",
    )
    p.add_argument(
        "--annotazioni",
        action="append",
        default=None,
        metavar="FOLDER",
        help="annotation delivery folder (FS-UCF-Crime or MIVIA-Crime), for the "
        "comparison row; repeatable. Without it, the local ones are tried",
    )
    p.add_argument(
        "--off-threshold",
        type=float,
        default=None,
        help="hysteresis CLOSING threshold; defaults to tau",
    )
    p.add_argument("--min-len", type=int, default=1, help="minimum interval length, in frames")
    p.add_argument("--fourcc", default="mp4v", help="codec of the annotated mp4")
    p.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    p.add_argument(
        "--attention",
        action="store_true",
        help="add the attention maps (attention.mp4 and attention_affiancato.mp4); requires --modo live",
    )
    p.add_argument("--no-timeline", action="store_true", help="do not write timeline.png")
    p.add_argument(
        "--no-copia-video",
        action="store_true",
        help="reference the video instead of copying it next to the page",
    )
    p.add_argument(
        "--elenca", action="store_true", help="list the videos the run can replay, and exit"
    )
    return p


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
