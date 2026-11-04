"""Progressive inference on a file played at 1x, through the shared engine.

The browser reports the video time: pause and buffering stop the arrival of frames.
The gate precedes decoding and forward, not only publication. Scores are also kept in
a journal separate from the official runs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

from evd.eval.demo_metrics import compute_timing, metriche_video
from evd.exp.demo import RunDemo, annotation, live_engine
from evd.infer.engine import VideoResult, video_result
from evd.types import WindowSpec


class DemoStream:
    """Append-only journal and player clock; a single process writes the scores."""

    def __init__(self, events: Path, clock: Path, *, timeout_s: float = 60) -> None:
        self.events, self.clock = events, clock
        self.timeout_s = timeout_s
        self.started = time.monotonic()
        self.compute_started = self.started
        self.events.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, **record) -> None:
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()

    def position(self) -> float:
        try:
            data = json.loads(self.clock.read_text(encoding="utf-8"))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            if time.monotonic() - self.started > self.timeout_s:
                raise TimeoutError("Player not connected. Reopen the live view.") from None
            return 0.0
        if time.time() - data["updated_at"] > self.timeout_s:
            raise TimeoutError("Player disconnected: analysis interrupted.")
        return float(data["position_s"])

    def wait_window(self, window: WindowSpec, fps: float) -> None:
        # Exclusive bound: frame end_frame must have been acquired.
        available = (window.end_frame + 1) / fps
        while self.position() < available:
            time.sleep(0.05)
        self.compute_started = time.monotonic()


def analyze_direct(
    run: RunDemo,
    video: Path,
    *,
    events: Path,
    clock: Path,
    device: str | None = None,
    backbone_checkpoint: str | Path | None = None,
    hop_s: float | None = None,
    window_s: float | None = None,
    timing: dict | None = None,
    annotations: str | Path | Sequence[str | Path] | None = None,
    off_threshold: float | None = None,
    min_len: int = 1,
) -> VideoResult:
    """Publish every score as soon as it is computed and return the usual final result."""
    channel = DemoStream(events, clock)
    channel.emit(type="loading", message="Loading the local model")
    engine, _ = live_engine(
        run,
        device=device,
        batch_size=1,
        backbone_checkpoint=backbone_checkpoint,
        hop_s=hop_s,
        window_s=window_s,
    )
    meta = engine.decoder.probe(video)
    windows = engine.sampler.windows_for_video(meta)
    truth = annotation(annotations, meta)
    timings = timing if timing is not None else {}
    costs = []
    channel.started = time.monotonic()
    channel.emit(
        type="ready",
        video_id=meta.video_id,
        duration_s=meta.num_frames / meta.fps,
        fps=meta.fps,
        threshold=run.threshold,
        total=len(windows),
        hop_s=hop_s if hop_s is not None else run.protocol["temporal"]["hop_s"],
        window_s=window_s if window_s is not None else run.protocol["temporal"].get("window_s"),
        metrics={
            k: run.metrics[k] for k in ("chunk_auc", "chunk_ap", "eval_split") if k in run.metrics
        }
        | {"video_f1": run.metrics.get("video_prf", {}).get("f1")},
        video_metrics=metriche_video(
            [],
            meta,
            run.threshold,
            annotation=truth,
            complete=False,
            threshold_source=run.threshold_source,
        ),
    )
    scores = []
    for item in engine.stream(
        video,
        meta,
        before_window=lambda w: channel.wait_window(w, meta.fps),
    ):
        scores.append(item)
        available = (item.window.end_frame + 1) / meta.fps
        compute_ms = (time.monotonic() - channel.compute_started) * 1000
        costs.append(compute_ms)
        timings.update(compute_timing(costs))
        # The window decision uses the run threshold; final intervals (hysteresis/causal
        # projection) remain the engine's responsibility.
        channel.emit(
            type="score",
            index=len(scores),
            window_id=item.window.window_id,
            start_s=item.window.frame_span[0] / meta.fps,
            end_s=available,
            score=float(item.anomaly),
            alarm=bool(item.anomaly >= run.threshold),
            compute_ms=compute_ms,
            observed_s=channel.position(),
            published_at=time.time(),
            video_metrics=metriche_video(
                scores,
                meta,
                run.threshold,
                annotation=truth,
                timing=timings,
                complete=False,
                threshold_source=run.threshold_source,
            ),
        )
    return video_result(
        engine.mapper,
        meta,
        scores,
        threshold=run.threshold,
        off_threshold=off_threshold,
        min_len=min_len,
    )
