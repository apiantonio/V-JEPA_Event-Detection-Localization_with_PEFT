"""``WindowScorer`` implementations.

Only :class:`HeadScorer` is implemented; ``PredictorErrScorer`` and ``TextSimScorer``
from the design are optional extensions that were never approved.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from evd.data.dataset import cached_output_from_batch
from evd.models.backbones.base import encode_multilevel
from evd.types import WindowScores


@runtime_checkable
class WindowScorer(Protocol):
    def score(self, batch: dict[str, Any]) -> list[WindowScores]: ...


def anomaly_from_logits(
    logits: torch.Tensor,
    *,
    batch_size: int,
    n_views: int | None = None,
    normal_idx: int | None = None,
    anomaly_from: str = "one_minus_normal",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Turn head logits into the anomaly score, following the protocol convention.

    This is the single definition of the score: sigmoid on a one-logit head (ADR-031) and
    ``anomaly = 1 - p(Normal)`` on a multi-class head (docs/03 §4.4), averaging views after
    the softmax as in the Meta eval path. It has two callers: :meth:`HeadScorer._from_logit`
    and the runner, which reuses validation logits to estimate the threshold (ADR-053).

    :param logits: head output, ``[B·V, C]``.
    :param batch_size: number of windows, views excluded.
    :param n_views: views per window, ``None`` for a single view.
    :param normal_idx: index of the ``Normal`` class; required only for multi-class heads.
    :param anomaly_from: ``one_minus_normal`` or ``max_event``.
    :return: ``(anomaly [B], probs [B, C])``; ``probs`` is ``None`` on a binary head.
    :raises ValueError: if the head does not produce ``[B, C]``, or ``normal_idx`` is
        missing or out of range.
    """
    if logits.ndim != 2:
        raise ValueError(f"the head must produce [B, C], not {tuple(logits.shape)}")
    if logits.shape[1] == 1:
        anomaly = torch.sigmoid(logits[:, 0])
        if n_views is not None:
            anomaly = anomaly.view(batch_size, n_views).mean(dim=1)
        return anomaly, None
    probs = F.softmax(logits, dim=-1)
    if n_views is not None:
        probs = probs.view(batch_size, n_views, -1).mean(dim=1)
    if normal_idx is None:
        raise ValueError(
            "normal_idx is required for a multi-class head; "
            "derive it from the class table of the manifest"
        )
    if not 0 <= normal_idx < probs.shape[1]:
        raise ValueError(f"normal_idx={normal_idx} outside the {probs.shape[1]} classes")
    if anomaly_from == "one_minus_normal":
        anomaly = 1.0 - probs[:, normal_idx]
    else:
        mask = [i for i in range(probs.shape[1]) if i != normal_idx]
        anomaly = probs[:, mask].max(dim=-1).values
    return anomaly, probs


class HeadScorer:
    """Backbone (+ adapter) + head: one score per window.

    The binary path is the active one (ADR-031): with a one-logit head the score is the
    sigmoid and ``normal_idx`` is not needed. The multi-class branch keeps the
    ``anomaly = 1 - p(Normal)`` convention; ``normal_idx`` has no default (13 on UCF-Crime,
    different on synthetic data).
    """

    def __init__(
        self,
        backbone: Any,
        head: Any,
        *,
        levels: tuple[int, ...] = (-1,),
        normal_idx: int | None = None,
        anomaly_from: str = "one_minus_normal",
        device: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        if anomaly_from not in ("one_minus_normal", "max_event"):
            raise ValueError(f"unknown anomaly_from: {anomaly_from!r}")
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"unknown precision: {precision!r}, expected 'fp32' or 'bf16'")
        self.backbone, self.head = backbone, head
        self.levels, self.normal_idx = tuple(levels), normal_idx
        self.anomaly_from, self.device = anomaly_from, device
        self.precision = precision
        """Numeric regime of the evaluation forward.

        The default ``fp32`` keeps the historical behaviour (training runs in AMP BF16);
        ``bf16`` is opt-in, see ``scripts/e0j_autocast.py``.
        """

    @torch.no_grad()
    def score(self, batch: dict[str, Any]) -> list[WindowScores]:
        if "features" in batch:
            return self._score_from_cache(batch)
        clips, batch_size, n_views = self.prepare_clips(batch)
        with self.autocast():
            feats = encode_multilevel(self.backbone, clips, self.levels)
            logits = self.head(feats)
        return self._from_logit(logits.float(), batch, batch_size=batch_size, n_views=n_views)

    def autocast(self) -> torch.autocast:
        """:return: the numeric regime shared by scoring and explanations."""
        return torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=self.precision == "bf16",
        )

    def prepare_clips(self, batch: dict[str, Any]) -> tuple[torch.Tensor, int, int | None]:
        """Prepare the views without disabling gradients.

        :param batch: windows already decoded and preprocessed by the dataset.
        :return: clips, batch size and number of views.
        """
        clips = batch["clip"].to(self.device)
        batch_size = clips.shape[0]
        n_views: int | None = None
        if clips.ndim == 6:
            # [B,V,C,T,H,W]: views are TTA of the same window, kept separate until the softmax and
            # averaged after the head, as in the Meta eval path.
            n_views = clips.shape[1]
            clips = clips.flatten(0, 1)
        elif clips.ndim != 5:
            raise ValueError(
                f"expected clips [B,C,T,H,W] or [B,V,C,T,H,W], not {tuple(clips.shape)}"
            )
        return clips, batch_size, n_views

    @torch.no_grad()
    def _score_from_cache(self, batch: dict[str, Any]) -> list[WindowScores]:
        """Compute the same score from precomputed features (frozen backbone only).

        The store rejects ``three_views``, so ``n_views`` is always ``None`` here.

        :raises ValueError: if the feature levels differ from those read by the scorer (same
            set and same order are required).
        """
        raw = batch.get("levels") or (batch["layer"],)
        expected = tuple(int(v[0]) if torch.is_tensor(v) and v.ndim else int(v) for v in tuple(raw))
        if self.levels != expected:
            raise ValueError(
                f"the scorer reads levels {list(self.levels)} but the cache serves features of "
                f"levels {list(expected)}: the cache keys were built with a different set "
                f"or order"
            )
        dtype = next(self.head.parameters()).dtype
        feats = cached_output_from_batch(batch, device=self.device, dtype=dtype)
        return self._from_logit(
            self.head(feats), batch, batch_size=feats.tokens.shape[0], n_views=None
        )

    def _from_logit(
        self,
        logits: torch.Tensor,
        batch: dict[str, Any],
        *,
        batch_size: int,
        n_views: int | None,
    ) -> list[WindowScores]:
        """Map ``[B, C]`` logits to per-window scores, shared by both paths.

        The formula lives in :func:`anomaly_from_logits`; this method only pairs scores with
        the batch windows.

        :param logits: head output, ``[B·V, C]``.
        :param batch: the batch, providing the windows.
        :param batch_size: number of windows, views excluded.
        :param n_views: views per window, ``None`` for a single view.
        :return: one :class:`~evd.types.WindowScores` per window.
        :raises ValueError: if the head does not produce ``[B, C]``, or ``normal_idx`` is
            missing or out of range.
        """
        anomaly, probs = anomaly_from_logits(
            logits,
            batch_size=batch_size,
            n_views=n_views,
            normal_idx=self.normal_idx,
            anomaly_from=self.anomaly_from,
        )
        windows: Sequence[Any] = batch["window"]
        return [
            WindowScores(
                window=w,
                anomaly=float(anomaly[i]),
                class_probs=None if probs is None else tuple(float(x) for x in probs[i]),
            )
            for i, w in enumerate(windows)
        ]
