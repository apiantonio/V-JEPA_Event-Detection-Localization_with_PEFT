"""Support and query sampling of the few-shot regime.

Used only by ``scripts/build_manifests.py``, never at runtime. No field of
:class:`SamplingRules` has a default: they are supervisor decisions. Monotonicity in k
is structural: for each (class, seed) a permutation is drawn once and increasing k
values extend the previous selection.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np

from evd.annotations.schema import TemporalAnnotation
from evd.protocol.manifest import (
    K_FULL,
    FewShotManifest,
    KBudget,
    ProtocolBudget,
    SupportItem,
    k_sort_key,
    normalize_k,
)
from evd.types import EventClass, VideoMeta
from evd.utils.logging import get_logger

log = get_logger(__name__)

DiversityRule = Literal["ceil_k_over_2", "one_per_video", "unlimited"]
NormalQuota = Literal["match_event_segments", "match_event_videos", "none"]
DedupRule = Literal["none", "sha256_representative"]
PoolExhausted = Literal["fail", "saturate"]

Segment = tuple[str, int, int, int]
"""``(video_id, start, end, class_idx)``: a support candidate."""


@dataclass(frozen=True)
class SamplingRules:
    """Protocol decisions, all required.

    ``k_values``               Q2, frozen {1, 2, 4, 8, 16, 32, ``full``}
    ``seeds``                  the same 5 at every k, docs/04 §2
    ``diversity``              §3.3, ``ceil_k_over_2``
    ``normal_quota``           §3.4, ``match_event_segments``
    ``window_cap_m``           maximum windows derived from each segment
    ``normal_segment_s``       required duration of a normal segment, in seconds
    ``dedup``                  Q19, the 21 byte-identical duplicates of the splits
    ``on_pool_exhausted``      behaviour when the normal quota exceeds the pool

    The last field exists for ``full``: with ``match_event_segments`` the full budget asks
    for as many normal segments as event segments (1554 against a pool of 781 on
    UCF-Crime). ``saturate`` fills up to the pool and logs both numbers; ``fail`` raises.
    """

    k_values: tuple[KBudget, ...]
    seeds: tuple[int, ...]
    diversity: DiversityRule
    normal_quota: NormalQuota
    window_cap_m: int
    normal_segment_s: float
    dedup: DedupRule
    on_pool_exhausted: PoolExhausted

    def __post_init__(self) -> None:
        if not self.k_values:
            raise ValueError("at least one k is required")
        object.__setattr__(self, "k_values", tuple(normalize_k(k) for k in self.k_values))
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError(f"k_values with repetitions: {self.k_values}")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.window_cap_m <= 0:
            raise ValueError(f"window_cap_m must be > 0, not {self.window_cap_m}")
        if self.on_pool_exhausted not in ("fail", "saturate"):
            raise ValueError(f"unrecognised on_pool_exhausted: {self.on_pool_exhausted!r}")
        if self.normal_quota != "none" and (
            not math.isfinite(self.normal_segment_s) or self.normal_segment_s <= 0
        ):
            raise ValueError(
                f"normal_segment_s must be finite and > 0, not {self.normal_segment_s}"
            )

    def ordered_k(self) -> list[KBudget]:
        """:return: the ``k`` values from the smallest to ``full``, always last."""
        return sorted(self.k_values, key=k_sort_key)

    def cap(self, k: KBudget) -> int | None:
        """Return the maximum number of segments from the same video at this k.

        :return: the cap, or ``None`` for no cap (the ``full`` budget).
        """
        if k == K_FULL:
            return None
        if self.diversity == "unlimited":
            return int(k)
        if self.diversity == "one_per_video":
            return 1
        return -(-int(k) // 2)  # ceil(k/2)

    def as_dict(self) -> dict[str, object]:
        return {
            "k_values": list(self.k_values),
            "seeds": list(self.seeds),
            "diversity": self.diversity,
            "normal_quota": self.normal_quota,
            "window_cap_m": self.window_cap_m,
            "normal_segment_s": self.normal_segment_s,
            "dedup": self.dedup,
            "on_pool_exhausted": self.on_pool_exhausted,
        }


@dataclass(frozen=True)
class ManifestBuilder:
    """Build manifests from an already loaded annotation source.

    ``train_videos`` and ``query_videos`` are the two pools and must be disjoint on input
    (docs/04 §3.2).
    """

    dataset: str
    protocol_version: str
    annotations: Mapping[str, TemporalAnnotation]
    classes: tuple[EventClass, ...]
    normal_idx: int
    train_videos: tuple[str, ...]
    query_videos: tuple[str, ...]
    inventory: Mapping[str, VideoMeta]
    rules: SamplingRules
    content_sha: Mapping[str, str] = field(default_factory=dict)
    """``video_id`` to sha256 from the F2 inventory; used only by ``dedup``."""
    annotation_source: str = ""
    builder_commit: str = ""
    created: str = ""

    def __post_init__(self) -> None:
        overlap = set(self.train_videos) & set(self.query_videos)
        if overlap:
            raise ValueError(
                f"train and query pools overlap on {len(overlap)} videos "
                f"(e.g. {sorted(overlap)[:3]}): they must be disjoint (docs/04 §3.2)"
            )
        if self.rules.dedup != "none" and not self.content_sha:
            raise ValueError(f"dedup={self.rules.dedup!r} requires content_sha from the inventory")

    # ------------------------------------------------------------------ #

    def build_all(self) -> list[FewShotManifest]:
        return [self.build(k, seed) for seed in self.rules.seeds for k in self.rules.ordered_k()]

    def build(self, k: KBudget, seed: int) -> FewShotManifest:
        k = normalize_k(k)
        if k not in self.rules.k_values:
            raise ValueError(f"k={k} is not among the protocol k_values {self.rules.k_values}")
        support_plan = self._support_plan(seed)
        support = support_plan[k]
        normal = self._normal_plan(seed, support_by_k=support_plan)[k]
        manifest = FewShotManifest(
            protocol_version=self.protocol_version,
            dataset=self.dataset,
            budget=ProtocolBudget(
                k=k,
                m=self.rules.window_cap_m,
                unit="annotated_event_segments_per_class",
            ),
            seed=seed,
            created=self.created or date.today().isoformat(),
            builder_commit=self.builder_commit,
            annotation_source=self.annotation_source,
            classes=self.classes,
            support=support,
            support_normal=normal,
            query=tuple(sorted(self.query_videos)),
        ).sealed()
        self._check(manifest)
        return manifest

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def _candidates(self) -> dict[int, list[Segment]]:
        """Return the annotated segments of the training pool, per class, after dedup."""
        pool = self._pool_after_dedup()
        out: dict[int, list[Segment]] = {
            c.idx: [] for c in self.classes if c.idx != self.normal_idx
        }
        for video_id in sorted(pool):
            ann = self.annotations.get(video_id)
            if ann is None:
                continue
            for start, end, class_idx in ann.intervals:
                if class_idx == self.normal_idx:
                    continue
                if class_idx not in out:
                    raise ValueError(f"{video_id}: class_idx {class_idx} outside the class table")
                out[class_idx].append((video_id, start, end, class_idx))
        return out

    def _pool_after_dedup(self) -> list[str]:
        """Apply Q19: the 21 byte-identical duplicates of the official splits.

        ``sha256_representative`` excludes from the support every training file whose content
        appears in the query set and keeps one representative per hash. The pair with
        discordant classes (``Assault050``/``Robbery138``) raises.
        """
        if self.rules.dedup == "none":
            return sorted(self.train_videos)

        query_hashes = {self.content_sha[v] for v in self.query_videos if v in self.content_sha}
        by_hash: dict[str, list[str]] = {}
        keep: list[str] = []
        for video_id in sorted(self.train_videos):
            digest = self.content_sha.get(video_id)
            if digest is None:
                keep.append(video_id)
                continue
            if digest in query_hashes:
                continue  # same content as a query video: it would contaminate the support
            by_hash.setdefault(digest, []).append(video_id)

        for digest, group in sorted(by_hash.items()):
            if len(group) > 1:
                labels = {self._classes_of(v) for v in group}
                if len(labels) > 1:
                    raise ValueError(
                        f"duplicates with discordant classes (sha {digest[:12]}): "
                        f"{sorted(group)} -> {sorted(labels)}. This is Q19: resolve it with the "
                        f"laboratory annotations, not by picking one."
                    )
            keep.append(group[0])
        return sorted(keep)

    def _classes_of(self, video_id: str) -> tuple[int, ...]:
        ann = self.annotations.get(video_id)
        return tuple(sorted({c for _, _, c in ann.intervals})) if ann else ()

    def _support_plan(self, seed: int) -> dict[KBudget, tuple[SupportItem, ...]]:
        """Return ``k`` to support, nested: the plan at k=2 contains the one at k=1.

        ``full`` comes last and takes all remaining candidates without diversity cap,
        extending the same permutation.
        """
        ordered_k = self.rules.ordered_k()
        candidates = self._candidates()
        per_class: dict[int, dict[KBudget, list[Segment]]] = {}

        for class_idx, segments in candidates.items():
            rng = np.random.default_rng([seed, class_idx])
            permuted = [segments[i] for i in rng.permutation(len(segments))]
            chosen: list[Segment] = []
            taken: Counter[str] = Counter()
            plan: dict[KBudget, list[Segment]] = {}
            for k in ordered_k:
                cap = self.rules.cap(k)
                quota = len(permuted) if k == K_FULL else int(k)
                for segment in permuted:
                    if len(chosen) >= quota:
                        break
                    if segment in chosen or (cap is not None and taken[segment[0]] >= cap):
                        continue
                    chosen.append(segment)
                    taken[segment[0]] += 1
                if len(chosen) < quota:
                    name = self._class_name(class_idx)
                    raise ValueError(
                        f"{name}: {quota} segments needed but the pool offers {len(chosen)} "
                        f"under the diversity constraint (cap {cap} per video, "
                        f"{len(segments)} segments over {len({s[0] for s in segments})} videos)"
                    )
                plan[k] = list(chosen)
            per_class[class_idx] = plan

        return {
            k: tuple(
                sorted(
                    (
                        SupportItem(
                            video_id,
                            self._interval_s(video_id, start, end),
                            class_idx,
                        )
                        for class_idx in sorted(per_class)
                        for video_id, start, end, class_idx in per_class[class_idx][k]
                    ),
                    key=lambda item: (item.class_idx, item.video_id, item.interval_s),
                )
            )
            for k in ordered_k
        }

    def _normal_plan(
        self, seed: int, *, support_by_k: Mapping[KBudget, tuple[SupportItem, ...]]
    ) -> dict[KBudget, tuple[SupportItem, ...]]:
        ordered_k = self.rules.ordered_k()
        if self.rules.normal_quota == "none":
            return dict.fromkeys(ordered_k, ())

        pool = [
            video_id
            for video_id in self._pool_after_dedup()
            if self._is_normal(video_id)
            and self._num_frames(video_id) >= self._normal_length_frames(video_id)
        ]
        if not pool:
            raise ValueError(
                "no normal video in the training pool is long enough for "
                f"{self.rules.normal_segment_s} s"
            )
        rng = np.random.default_rng([seed, self.normal_idx])
        permuted = [pool[i] for i in rng.permutation(len(pool))]
        lengths = {video_id: self._normal_length_frames(video_id) for video_id in permuted}
        starts = {
            video_id: int(rng.integers(0, self._num_frames(video_id) - lengths[video_id] + 1))
            for video_id in permuted
        }

        chosen: list[str] = []
        plan: dict[KBudget, tuple[SupportItem, ...]] = {}
        for k in ordered_k:
            support = support_by_k[k]
            wanted = (
                len(support)
                if self.rules.normal_quota == "match_event_segments"
                else len({item.video_id for item in support})
            )
            if wanted > len(permuted):
                if self.rules.on_pool_exhausted == "fail":
                    raise ValueError(
                        f"k={k}: {wanted} normal segments needed, the pool offers {len(permuted)}"
                    )
                # Declared saturation: the manifest is unbalanced and shows it (len(support) against
                # len(support_normal)).
                log.warning(
                    "k=%s: normal quota saturated - %d segments requested, pool %d; "
                    "event:normal ratio in the manifest is %d:%d",
                    k,
                    wanted,
                    len(permuted),
                    len(support),
                    len(permuted),
                )
                wanted = len(permuted)
            for video_id in permuted:
                if len(chosen) >= wanted:
                    break
                if video_id not in chosen:
                    chosen.append(video_id)
            plan[k] = tuple(
                sorted(
                    (
                        SupportItem(
                            v,
                            self._interval_s(v, starts[v], starts[v] + lengths[v] - 1),
                            self.normal_idx,
                        )
                        for v in chosen
                    ),
                    key=lambda item: (item.video_id, item.interval_s),
                )
            )
        return plan

    # ------------------------------------------------------------------ #

    def _check(self, manifest: FewShotManifest) -> None:
        """Check the rules of docs/04 §3 on the finished manifest."""
        query = set(manifest.query)
        contaminated = manifest.support_videos() & query
        if contaminated:
            raise ValueError(f"videos in both support and query: {sorted(contaminated)[:5]}")

        per_class = Counter(item.class_idx for item in manifest.support)
        if manifest.k != K_FULL:
            # At ``full`` the per-class quota is not constant by definition (docs/12 Q2);
            # `_support_plan` guarantees that no candidate is left out.
            wrong = {self._class_name(idx): n for idx, n in per_class.items() if n != manifest.k}
            if wrong:
                raise ValueError(f"k={manifest.k} but the per-class quotas do not match: {wrong}")

        cap = self.rules.cap(manifest.k)
        if cap is not None:
            per_video = Counter((item.class_idx, item.video_id) for item in manifest.support)
            over = {key: n for key, n in per_video.items() if n > cap}
            if over:
                raise ValueError(f"diversity constraint violated (cap {cap}): {over}")

        for item in (*manifest.support, *manifest.support_normal):
            duration = self._duration_s(item.video_id)
            if item.interval_s[1] > duration + 1e-9:
                raise ValueError(
                    f"{item.video_id}: segment {item.interval_s} beyond {duration:.9f} s"
                )

    def _class_name(self, class_idx: int) -> str:
        for c in self.classes:
            if c.idx == class_idx:
                return c.name
        return str(class_idx)

    def _is_normal(self, video_id: str) -> bool:
        ann = self.annotations.get(video_id)
        return ann is not None and ann.is_normal

    def _num_frames(self, video_id: str) -> int:
        meta = self.inventory.get(video_id)
        return meta.num_frames if meta else 0

    def _meta(self, video_id: str) -> VideoMeta:
        meta = self.inventory.get(video_id)
        if meta is None:
            raise ValueError(f"{video_id}: video metadata missing from the inventory")
        if not math.isfinite(meta.fps) or meta.fps <= 0:
            raise ValueError(f"{video_id}: invalid fps ({meta.fps})")
        return meta

    def _duration_s(self, video_id: str) -> float:
        meta = self._meta(video_id)
        return meta.num_frames / meta.fps

    def _interval_s(self, video_id: str, start: int, end: int) -> tuple[float, float]:
        """Convert the native ``[start, end]`` into ``[start_s, end_s)``."""
        meta = self._meta(video_id)
        if start < 0 or end < start or end >= meta.num_frames:
            raise ValueError(
                f"{video_id}: frame interval ({start}, {end}) outside {meta.num_frames}"
            )
        return (start / meta.fps, (end + 1) / meta.fps)

    def _normal_length_frames(self, video_id: str) -> int:
        meta = self._meta(video_id)
        return max(1, int(math.floor(self.rules.normal_segment_s * meta.fps + 0.5)))


def check_monotonic(manifests: Sequence[FewShotManifest]) -> None:
    """Check that the support at k contains the one at k' < k, for the same seed
    (docs/04 §3.5).

    :raises ValueError: if a smaller support is not contained in a larger one.
    """
    by_seed: dict[int, list[FewShotManifest]] = {}
    for m in manifests:
        by_seed.setdefault(m.seed, []).append(m)
    for seed, group in by_seed.items():
        group.sort(key=lambda m: k_sort_key(m.k))
        for previous, current in zip(group, group[1:], strict=False):
            missing = set(previous.support) - set(current.support)
            if missing:
                raise ValueError(
                    f"seed {seed}: the support at k={current.k} does not contain the one at "
                    f"k={previous.k} ({len(missing)} segments lost)"
                )
