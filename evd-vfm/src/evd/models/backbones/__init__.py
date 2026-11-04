"""Video backbone wrappers; importing this package registers them.

``tiny`` (toy) and ``vjepa21`` (built from the Meta source, ``strict=True``).
"""

from evd.models.backbones.base import BACKBONES, VideoBackbone
from evd.models.backbones.tiny import TinyBackbone
from evd.models.backbones.vjepa21 import VJEPA21Backbone, vjepa21_backbone

__all__ = [
    "BACKBONES",
    "TinyBackbone",
    "VJEPA21Backbone",
    "VideoBackbone",
    "vjepa21_backbone",
]
