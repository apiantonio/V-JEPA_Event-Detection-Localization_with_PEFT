"""Explainability: what the model looks at when it decides (ADR-063).

This package computes; drawing belongs to ``apps/demo/`` (ADR-015), which does not
import ``torch``.

* :mod:`~evd.explain.attention`: reads the pooler cross-attention weights with the
  fidelity check that makes them credible, and decides how they are displayed
  (:data:`~evd.explain.attention.CONTRAST_MODES`, since the raw map is almost flat);
* :mod:`~evd.explain.project`: projects window maps onto native frames with the same
  rule used for scores.
"""

from evd.explain.attention import (
    CONTRAST_MODES,
    FIDELITY_TOLERANCE,
    AttentionMap,
    Contrast,
    MapStatistics,
    capture_attention,
    contrast_of,
    cross_attention_module,
    map_statistics,
    reduce_over_frames,
)
from evd.explain.project import WindowsPerFrame, map_per_frame, windows_per_frame

__all__ = [
    "CONTRAST_MODES",
    "FIDELITY_TOLERANCE",
    "Contrast",
    "WindowsPerFrame",
    "AttentionMap",
    "MapStatistics",
    "capture_attention",
    "contrast_of",
    "windows_per_frame",
    "map_per_frame",
    "cross_attention_module",
    "reduce_over_frames",
    "map_statistics",
]
