"""Classification heads on top of the backbone tokens.

``LinearProbe`` (collapse diagnostics, smoke head), ``AttentiveProbe``,
``DetectionHead`` (and demonstrator) and the class-query heads of the class-routing
localization campaign (``class_queries``).
"""

from evd.models.heads.attentive import AttentiveProbe
from evd.models.heads.base import HEADS, Head
from evd.models.heads.class_queries import (
    BinaryAndClassHead,
    ClassQueryBranchHead,
    ClassQueryHead,
)
from evd.models.heads.detection import DetectionHead
from evd.models.heads.linear import LinearProbe
from evd.models.heads.pooling import AttentivePooling

__all__ = [
    "HEADS",
    "AttentivePooling",
    "AttentiveProbe",
    "BinaryAndClassHead",
    "ClassQueryBranchHead",
    "ClassQueryHead",
    "DetectionHead",
    "Head",
    "LinearProbe",
]
