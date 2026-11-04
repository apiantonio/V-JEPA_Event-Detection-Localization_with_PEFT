#!/usr/bin/env python
r"""Attention maps of a run on a video. Thin CLI (ADR-015).

    # with the defaults: the best model and a test video, no arguments
    python evd-vfm/scripts/explain_attention.py

    # choosing model and video
    python evd-vfm/scripts/explain_attention.py \
        --run runs/e3r_ucf_vjepa21l_lora-r32-qkvomlp_s42 \
        --video ../data/raw/Fighting/Fighting047_x264.mp4

    # in bulk, on the videos chosen by scripts/scelta_video_explain.py
    python evd-vfm/scripts/explain_attention.py --selezione runs/analisi_explain/selezione.json

Writes to ``--out`` (default ``runs/{run_id}/explain/{video_id}/``):

* ``attention.mp4``: the map over the video, for figures and slides;
* ``attention_affiancato.mp4``: original on the left, map on the right, to judge the map;
* ``attention_mappa.mp4``: the heatmap alone (``--layout tutti``), telling whether a
  barely visible map is weak or just covered by the video;
* ``attention.npz``: the raw, non-normalized maps;
* ``attention.json``: provenance, scores, the fidelity gap of the recomputation, and the
  ``contrasto`` (how the image was stretched) and ``concentrazione`` (how concentrated the
  map was before stretching) blocks.

The pooler map is almost flat, and that is not a failure (on ``Arson016_x264`` weights lie
between 0.60x and 1.70x the uniform weight, entropy 0.997-0.9999): hence the default
``percentile`` contrast and ``--contrasto scarto``, which subtracts the video mean map.
A checkpoint is required: the maps are an internal state of the forward pass, so the
demonstrator ``replay`` mode cannot have them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from apps.demo import attention as render  # noqa: E402

from evd.exp.demo import open_run  # noqa: E402
from evd.exp.explain import analyze_attention, write_artifacts  # noqa: E402
from evd.explain.attention import CONTRAST_MODES  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("explain_attention")

#: Default run: the best matrix cell by chunk-AUC among those with a local checkpoint.
RUN_DEFAULT = "runs/e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42"

#: Default video: an anomalous test video available locally and already used to test the
#: demonstrator (ADR-055).
VIDEO_DEFAULT = "../data/raw/Arson/Arson016_x264.mp4"

#: File name of every cut. ``attention.mp4`` stays the overlay: the name already written
#: artifacts carry and ``docs/16`` cites.
CUT_NAMES = {
    "overlay": "attention.mp4",
    "affiancato": "attention_affiancato.mp4",
    "mappa": "attention_mappa.mp4",
}

#: Where to look for the backbone weights when the run config points to the cluster.
LOCAL_BACKBONE = "../data/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"


def main(argv: list[str] | None = None) -> int:
    """Run the attention analysis.

    :param argv: the arguments, ``None`` for ``sys.argv``.
    :return: the process exit code.
    """
    args = _parser().parse_args(argv)
    setup_logging()

    video = _video_list(args)
    if not video:
        print("no video to analyse", file=sys.stderr)
        return 2

    run = open_run(args.run)
    backbone = args.backbone_checkpoint
    if backbone is None and (_ROOT / LOCAL_BACKBONE).exists():
        backbone = str((_ROOT / LOCAL_BACKBONE).resolve())

    failed: list[tuple[str, str]] = []
    for i, path in enumerate(video, start=1):
        log.info("[%d/%d] %s", i, len(video), path.name)
        try:
            _one_video(run, path, args, backbone)
        except Exception as error:  # a broken video does not stop the batch
            if len(video) == 1:
                raise
            log.error("%s: %s", path.name, error)
            failed.append((path.name, str(error)))

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(video)} videos did not pass:", file=sys.stderr)
        for name, error in failed:
            print(f"  {name}: {error}", file=sys.stderr)
        return 1
    return 0


def _one_video(run, percorso: Path, args: argparse.Namespace, backbone: str | None) -> None:
    """Analyse a video and write its artifacts.

    :param run: the open run.
    :param percorso: the video file.
    :param args: the CLI arguments.
    :param backbone: the backbone weights, or ``None`` for those of the config.
    :raises RuntimeError: if the fidelity check fails (a map not matching the model
        computation is not saved).
    """
    result = analyze_attention(
        run,
        percorso,
        device=args.device,
        backbone_checkpoint=backbone,
        batch_size=args.batch_size,
        hop_s=args.hop_s,
        contrast_mode=args.contrasto,
    )
    if not result.fedele:
        raise RuntimeError(
            f"the attention recomputation deviates by {result.scarto_fedelta_max:.3g} "
            f"from the pooler output ({result.dtype_forward}): the map is NOT what the "
            f"model computed and is not saved. This is the ADR-063 check"
        )
    out = Path(args.out) if args.out else run.dir / "explain" / result.video_id
    written = write_artifacts(result, out)
    log.info(
        "%s: %d windows, fidelity gap %.3g (%s), mass on prompts %.4f",
        result.video_id,
        len(result.finestre),
        result.scarto_fedelta_max,
        result.dtype_forward,
        result.massa_prompt_max,
    )
    lo, hi = result.contrasto.estremi_in_uniforme()
    for cut in _cuts(args.layout):
        written.append(
            render.scrivi(
                percorso,
                out / CUT_NAMES[cut],
                heat=result.heat,
                anomaly=result.anomaly_per_frame,
                threshold=result.threshold,
                fps=result.fps,
                video_id=result.video_id,
                run_id=result.run_id,
                layout=cut,
                fourcc=args.fourcc,
                alfa=args.alfa,
                gamma=args.gamma,
                divergente=result.contrasto.divergente,
                scale_lo=lo,
                scale_hi=hi,
                contrast_mode=result.contrasto.modo,
            )
        )
    print(f"\n{result.video_id} -> {out}")
    for p in written:
        print(f"  {p.name}")


def _cuts(layout: str) -> tuple[str, ...]:
    """:param layout: the ``--layout`` value.

    :return: the cuts to write. ``entrambi`` stays the historical default (overlay and
        side by side) and ``tutti`` adds the bare map.
    """
    if layout == "entrambi":
        return ("overlay", "affiancato")
    if layout == "tutti":
        return ("overlay", "affiancato", "mappa")
    return (layout,)


def _video_list(args: argparse.Namespace) -> list[Path]:
    """Return the videos to analyse, from ``--video`` or ``--selezione``.

    :param args: the CLI arguments.
    :return: existing paths, in order and without repetitions; missing ones are reported
        and skipped.
    """
    if args.selezione:
        data = json.loads(Path(args.selezione).read_text(encoding="utf-8"))
        # The selection is per (video, class), so a multi-class video appears once per class:
        # without deduplication the same forward pass would be repeated.
        raw = list(dict.fromkeys(Path(v["percorso"]) for v in data["video"] if v.get("percorso")))
    else:
        raw = [Path(args.video)]
    present: list[Path] = []
    for p in raw:
        absolute = p if p.is_absolute() else (_ROOT / p)
        if not absolute.exists():
            log.warning("%s does not exist: skipping it", absolute)
        elif absolute.resolve() not in present:
            present.append(absolute.resolve())
    return present


def _parser() -> argparse.ArgumentParser:
    """:return: the CLI parser."""
    p = argparse.ArgumentParser(
        description="Pooler attention maps: where the model looks to decide.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run", default=RUN_DEFAULT, help="the runs/{run_id}/ folder with the weights")
    p.add_argument("--video", default=VIDEO_DEFAULT, help="the video to analyse")
    p.add_argument(
        "--selezione",
        default=None,
        help="a scelta_video_explain.py JSON: analyse every video it names",
    )
    p.add_argument(
        "--out", default=None, help="output folder (default: runs/{run_id}/explain/{video_id})"
    )
    p.add_argument("--layout", choices=render.LAYOUT, default="entrambi", help="which mp4 to write")
    p.add_argument(
        "--contrasto",
        choices=CONTRAST_MODES,
        default="percentile",
        help="how the raw map becomes an image; 'scarto' subtracts the video mean "
        "map and shows only the content-dependent part, 'massimo' is the "
        "unstretched scale (readable only if the pooler is selective), 'finestra' hides "
        "windows where the model looks everywhere",
    )
    p.add_argument(
        "--alfa",
        type=float,
        default=render.ALFA,
        help="map opacity at the top of the scale",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=render.GAMMA,
        help="exponent on the already contrasted intensity; below 1 lifts mid values",
    )
    p.add_argument("--device", default=None, help="where the forward pass runs")
    p.add_argument("--batch-size", type=int, default=1, help="windows per forward pass")
    p.add_argument("--hop-s", type=float, default=None, help="grid step, to densify")
    p.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="backbone weights, if not where the run had them",
    )
    p.add_argument("--fourcc", default="mp4v", help="mp4 codec")
    return p


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
