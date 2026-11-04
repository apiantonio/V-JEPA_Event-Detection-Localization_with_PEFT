"""Temporal annotations: schema, validator and per-format adapters.

Every adapter exposes two stages: ``parse()`` returns tolerant
``list[RawInterval]`` rows (broken rows included, so the validator can report
them) and ``load()`` returns strict ``dict[str, TemporalAnnotation]``.
"""

from evd.annotations.schema import (
    AnnotationSource,
    RawInterval,
    TemporalAnnotation,
    frame_class_labels,
    frame_labels,
    materialize,
    window_label,
)
from evd.annotations.validate import Finding, ValidationReport, validate

__all__ = [
    "AnnotationSource",
    "Finding",
    "RawInterval",
    "TemporalAnnotation",
    "ValidationReport",
    "frame_class_labels",
    "frame_labels",
    "materialize",
    "validate",
    "window_label",
]
