"""Experimental protocol: immutable files and the rules that generate them.

Two regimes coexist. The active one is fully supervised:
:mod:`evd.protocol.split` declares the video partitions and :mod:`evd.protocol.chunks`
derives the chunk grid with labels and weights. The previous few-shot regime
(:mod:`evd.protocol.manifest`, :mod:`evd.protocol.builder`) is no longer the
production path and stays only so that ``fsp_v1`` manifests remain readable and their
runs reproducible.

Invariant 1: a protocol file consumed by a run is never rewritten; a different value
is a new version, with a new file and a changelog.
"""

from evd.protocol.builder import ManifestBuilder, SamplingRules
from evd.protocol.chunks import (
    ChunkItem,
    ChunkSummary,
    chunks_for_video,
    enumerate_chunks,
    riepilogo,
    sampler_for_protocol,
)
from evd.protocol.manifest import (
    FewShotManifest,
    ProtocolBudget,
    SupportItem,
    load_manifest,
    manifest_sha256,
    save_manifest,
)
from evd.protocol.split import SplitSpec, load_split, save_split, split_sha256
from evd.protocol.split_builder import (
    SplitReport,
    ValidationRules,
    build_split,
    stratum_of,
)

__all__ = [
    # --- active regime (edp, ADR-031) ---
    "ChunkItem",
    "ChunkSummary",
    "SplitReport",
    "SplitSpec",
    "ValidationRules",
    "build_split",
    "chunks_for_video",
    "enumerate_chunks",
    "load_split",
    "riepilogo",
    "sampler_for_protocol",
    "save_split",
    "split_sha256",
    "stratum_of",
    # --- historical few-shot regime: read, do not produce ---
    "FewShotManifest",
    "ManifestBuilder",
    "ProtocolBudget",
    "SamplingRules",
    "SupportItem",
    "load_manifest",
    "manifest_sha256",
    "save_manifest",
]
