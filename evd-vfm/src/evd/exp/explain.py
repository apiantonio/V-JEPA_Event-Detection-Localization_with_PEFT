"""Explainability orchestrator: from (run, video) to attention maps (ADR-063).

Lives here, not in ``apps/``, because apps do not build models (ADR-015): this module
builds the run model, streams its windows and collects the attention weights;
:mod:`apps.demo.attention` draws, receiving arrays only. Windows come from the engine
(invariant 4) and the maps/windows pairing is checked by counting. Live mode is
required: attention weights are internal forward state, not saved scores.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evd.exp.demo import RunDemo, live_engine, open_run
from evd.explain.attention import (
    FIDELITY_TOLERANCE,
    Contrast,
    MapStatistics,
    capture_attention,
    contrast_of,
    map_statistics,
)
from evd.explain.project import map_per_frame, windows_per_frame
from evd.types import VideoMeta, WindowSpec
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = ["AttentionResult", "analyze_attention", "write_artifacts"]


@dataclass(frozen=True)
class AttentionResult:
    """Everything needed to draw, and nothing else.

    :param heat: ``[num_frames, H', W']`` float32 in ``[0, 1]``, after the
        :class:`~evd.explain.attention.Contrast`; ``nan`` on frames covered by no window
        (with the causal policy, the first ones).
    :param anomaly_per_frame: ``[num_frames]`` float32, the score projected with the same
        rule.
    :param mappe: ``[n_windows, T', H', W']`` float32, unnormalized, for the artifact on
        disk.
    :param contrasto: transformation applied to ``heat``, with its extremes (for the
        colour bar).
    :param statistiche: concentration of the map before the contrast stretch.
    """

    run_id: str
    video_id: str
    grid: tuple[int, int, int]
    finestre: tuple[WindowSpec, ...]
    anomaly: tuple[float, ...]
    mappe: np.ndarray
    heat: np.ndarray
    anomaly_per_frame: np.ndarray
    coperto: np.ndarray
    num_frames: int
    fps: float
    threshold: float
    threshold_source: str
    scarto_fedelta_max: float
    massa_prompt_max: float
    dtype_forward: str
    n_teste: int
    contrasto: Contrast
    statistiche: MapStatistics

    @property
    def fedele(self) -> bool:
        """:return: whether the worst gap is within the tolerance of its dtype
        (:data:`~evd.explain.attention.FIDELITY_TOLERANCE`).
        """
        import torch

        limit = FIDELITY_TOLERANCE.get(getattr(torch, self.dtype_forward, torch.float32), 1e-4)
        return bool(np.isfinite(self.scarto_fedelta_max)) and self.scarto_fedelta_max <= limit


def analyze_attention(
    run: RunDemo,
    video: str | Path,
    *,
    device: str | None = None,
    backbone_checkpoint: str | Path | None = None,
    batch_size: int = 1,
    hop_s: float | None = None,
    window_s: float | None = None,
    contrast_mode: str = "percentile",
    percentili: tuple[float, float] = (1.0, 99.0),
) -> AttentionResult:
    """Compute scores and attention maps of a video, with the same engine.

    :param run: the open run (``open_run``).
    :param video: video file to analyse.
    :param device: forward device; ``None`` uses the run one.
    :param backbone_checkpoint: backbone weights, if not where the run had them.
    :param batch_size: windows per forward; 1 by default (an 8 GiB GPU holds ViT-L at 9216
        tokens one window at a time).
    :param hop_s: grid step, to densify the map; ``None`` uses the protocol one. Changing
        it changes the windows, as the artifact declares.
    :param window_s: window duration; same caveats.
    :param contrast_mode: how the raw map becomes an image, one of
        :data:`~evd.explain.attention.CONTRAST_MODES`; ``"percentile"`` by default, since
        with ``"massimo"`` the almost flat softmax gives a uniform image.
    :param percentili: extremes of the ``percentile`` mode.
    :return: the result, ready to draw and write.
    :raises RuntimeError: if the number of maps differs from the number of windows.
    :raises ValueError: if the video produces no window.
    """
    engine, _spec = live_engine(
        run,
        device=device,
        batch_size=batch_size,
        backbone_checkpoint=backbone_checkpoint,
        hop_s=hop_s,
        window_s=window_s,
    )
    path = Path(video)
    meta: VideoMeta = engine.decoder.probe(path)
    log.info("%s: %d frame a %.3f fps", meta.video_id, meta.num_frames, meta.fps)

    with capture_attention(engine.scorer.head) as maps:
        scores = list(engine.stream(path, meta))

    if not scores:
        raise ValueError(
            f"{meta.video_id} produces no window with the run grid: "
            f"it is shorter than a window, or hop_s/window_s emptied it"
        )
    if len(maps) != len(scores):
        raise RuntimeError(
            f"{len(maps)} maps for {len(scores)} windows on {meta.video_id}: the hook and the "
            f"engine are not looking at the same sequence, so every score/map pairing "
            f"would be silently shifted"
        )

    windows = tuple(p.window for p in scores)
    anomaly = tuple(float(p.anomaly) for p in scores)
    assignment = windows_per_frame(engine.mapper, meta, windows)
    contrast = contrast_of(list(maps), modo=contrast_mode, percentili=percentili)
    statistics = map_statistics(list(maps))
    log.info(
        "attention %.2f-%.2f x uniform, entropy %.4f-%.4f, top-1%% %.4f; contrast %s on "
        "%.2f-%.2f x uniform",
        statistics.minimo_su_uniforme,
        statistics.massimo_su_uniforme,
        statistics.entropia_minima,
        statistics.entropia_mediana,
        statistics.massa_top1,
        contrast.modo,
        *contrast.estremi_in_uniforme(),
    )

    t_grid, h_grid, w_grid = maps[0].grid
    heat = np.full((meta.num_frames, h_grid, w_grid), np.nan, dtype=np.float32)
    per_frame = np.full(meta.num_frames, np.nan, dtype=np.float32)
    for f in range(meta.num_frames):
        slice_ = map_per_frame(maps, windows, assignment, f)
        if slice_ is None:
            continue
        heat[f] = contrast.apply_to(slice_)
        per_frame[f] = anomaly[int(assignment.indice[f])]

    return AttentionResult(
        run_id=run.run_id,
        video_id=meta.video_id,
        grid=(t_grid, h_grid, w_grid),
        finestre=windows,
        anomaly=anomaly,
        mappe=np.stack([m.pesi for m in maps]).astype(np.float32),
        heat=heat,
        anomaly_per_frame=per_frame,
        coperto=assignment.coperto,
        num_frames=meta.num_frames,
        fps=float(meta.fps),
        threshold=float(run.threshold),
        threshold_source=str(run.threshold_source),
        scarto_fedelta_max=max(m.scarto_fedelta for m in maps),
        massa_prompt_max=max(m.massa_prompt for m in maps),
        dtype_forward=maps[0].dtype,
        n_teste=maps[0].n_teste,
        contrasto=contrast,
        statistiche=statistics,
    )


def write_artifacts(risultato: AttentionResult, out: Path) -> list[Path]:
    """Write the raw maps and their provenance next to the videos.

    Arrays go to a compressed ``.npz`` and metadata to a ``.json`` readable without NumPy.
    Maps are saved unnormalized, so the drawing normalization can change without rerunning
    the forward.

    :param risultato: output of :func:`analyze_attention`.
    :param out: destination folder; it is created.
    :return: the written files.
    """
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "attention.npz"
    np.savez_compressed(
        npz,
        mappe=risultato.mappe,
        heat=risultato.heat,
        anomaly_per_frame=risultato.anomaly_per_frame,
        coperto=risultato.coperto,
    )
    meta: dict[str, Any] = {
        "run_id": risultato.run_id,
        "video_id": risultato.video_id,
        "modo": "live",
        "grid": list(risultato.grid),
        "n_finestre": len(risultato.finestre),
        "num_frames": risultato.num_frames,
        "fps": risultato.fps,
        "threshold": risultato.threshold,
        "threshold_source": risultato.threshold_source,
        "anomaly": list(risultato.anomaly),
        "endpoint_finestre": [w.end_frame for w in risultato.finestre],
        "contrasto": {
            "modo": risultato.contrasto.modo,
            "lo": risultato.contrasto.lo,
            "hi": risultato.contrasto.hi,
            "uniforme": risultato.contrasto.uniforme,
            "estremi_in_uniforme": list(risultato.contrasto.estremi_in_uniforme()),
            "divergente": risultato.contrasto.divergente,
        },
        "concentrazione": {
            "minimo_su_uniforme": risultato.statistiche.minimo_su_uniforme,
            "massimo_su_uniforme": risultato.statistiche.massimo_su_uniforme,
            "entropia_minima": risultato.statistiche.entropia_minima,
            "entropia_mediana": risultato.statistiche.entropia_mediana,
            "massa_top1": risultato.statistiche.massa_top1,
            "scarto_su_deviazione": risultato.statistiche.scarto_su_deviazione,
        },
        "fedelta": {
            "scarto_massimo": risultato.scarto_fedelta_max,
            "dtype_forward": risultato.dtype_forward,
            "fedele": risultato.fedele,
        },
        "massa_prompt_massima": risultato.massa_prompt_max,
        "n_teste_mediate": risultato.n_teste,
        "avvertenza": (
            "Weights come from the pooler cross-attention, after three "
            "self-attention blocks: a high weight is a POSITION the query reads from, not "
            "a pixel that caused the decision alone. The image is STRETCHED by the "
            "contrast recorded here: the actual concentration is given by the "
            "'concentrazione' fields, and an entropy close to 1.0 means the pooler is "
            "doing almost a global average."
        ),
    }
    json_path = out / "attention.json"
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return [npz, json_path]


def open_(run_dir: str | Path) -> RunDemo:
    """:param run_dir: the ``runs/{run_id}/`` folder.
    :return: the open run; re-exports :func:`evd.exp.demo.open_run` for explainability
        scripts.
    """
    return open_run(run_dir)
