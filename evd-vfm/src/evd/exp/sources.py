"""From a ``configs/data/*.yaml`` group to videos, classes and annotations.

An experiment needs to know where files are, their frame count and rate, and the ground
truth; manifests and splits carry only identifiers, so they stay valid across machines
(invariant 1, ADR-001). This single bridge is shared by the manifest builder and the
runner, so protocols are built and executed on the same inventory.

Datasets are added by registration (a function plus a decorator, no ``if`` in the
runner). It lives in ``evd.exp`` because it composes ``data`` and ``annotations`` and
uses the ``models`` registry (docs/02 §5).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evd.annotations.schema import TemporalAnnotation
from evd.annotations.ucf_classes import normal_index, ucf_classes
from evd.annotations.ucf_lab import MIVIA_CRIME, UcfLabAnnotations
from evd.data.inventory import load_split_paths, read_records
from evd.models.registry import Registry
from evd.types import EventClass, VideoMeta
from evd.utils.hashing import file_sha256
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DATASOURCES",
    "DatasetBundle",
    "build_dataset",
]


@dataclass(frozen=True)
class DatasetBundle:
    """What a dataset provides, whichever it is.

    :param dataset_id: the ``dataset`` field of the ``run_id`` (``ucf``, ``synth``...).
    :param name: full name, for messages and metadata.
    :param classes: canonical catalogue, ``Normal`` included with the right index.
    :param normal_idx: index of ``Normal``; always explicit (13 on UCF-Crime, ADR-024
        point 7).
    :param inventory: ``video_id`` to metadata with resolved absolute path.
    :param annotations: ``video_id`` to ground truth.
    :param train_videos: training pool.
    :param query_videos: evaluation pool, disjoint from the training one.
    :param annotation_source: ``<adapter>@<sha>``, binding manifests and splits to the exact
        annotation version (docs/03 §5, ADR-022).
    :param content_sha: ``video_id`` to file sha256, when the inventory knows it (used only
        by the Q19 dedup rule).
    """

    dataset_id: str
    name: str
    classes: tuple[EventClass, ...]
    normal_idx: int
    inventory: Mapping[str, VideoMeta]
    annotations: Mapping[str, TemporalAnnotation]
    train_videos: tuple[str, ...]
    query_videos: tuple[str, ...]
    annotation_source: str
    content_sha: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        common = set(self.train_videos) & set(self.query_videos)
        if common:
            raise ValueError(
                f"{self.name}: {len(common)} videos are both in train and query "
                f"({sorted(common)[:3]}...). Disjointness is a protocol property "
                f"(docs/04 §3.2), not a sampling side effect"
            )
        without_meta = sorted(
            v for v in (*self.train_videos, *self.query_videos) if v not in self.inventory
        )
        if without_meta:
            raise ValueError(
                f"{self.name}: {len(without_meta)} split videos are not "
                f"in the inventory ({without_meta[:3]}...)"
            )
        if not 0 <= self.normal_idx < len(self.classes):
            raise ValueError(
                f"{self.name}: normal_idx {self.normal_idx} outside the "
                f"{len(self.classes)} classes of the catalogue"
            )

    def meta(self, video_id: str) -> VideoMeta:
        """:raises KeyError: if the video is not in the inventory, naming it."""
        try:
            return self.inventory[video_id]
        except KeyError:
            raise KeyError(
                f"{video_id} is not in the {self.name} inventory: do the manifest and the "
                f"inventory come from two different executions?"
            ) from None

    def n_classes(self) -> int:
        """:return: head outputs: event classes plus ``Normal``."""
        return len(self.classes)


DATASOURCES: Registry[Callable[[Mapping[str, Any]], DatasetBundle]] = Registry("datasource")


def build_dataset(cfg: Mapping[str, Any]) -> DatasetBundle:
    """Build the bundle described by the config ``data`` group.

    :param cfg: resolved ``data`` group; ``name`` selects the source.
    :return: the bundle, with absolute paths.
    :raises KeyError: if the dataset is not registered.
    :raises ValueError: if ``name`` is missing.
    """
    if "name" not in cfg:
        raise ValueError("the data group does not declare 'name': the source cannot be chosen")
    return DATASOURCES.get(str(cfg["name"]))(cfg)


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #

#: The synthetic split alternates videos within each class (first to training, second
#: to queries, ...), so every class has positives among the queries; defined in one
#: place, shared by builder and runner.


@DATASOURCES.register("synthetic")
def _synthetic(cfg: Mapping[str, Any]) -> DatasetBundle:
    """Return the synthetic world of docs/08 §3, regenerated from the seed.

    Videos live in the cache (``$EVD_CACHE_DIR``), not in the repository.

    :param cfg: the ``configs/data/synthetic.yaml`` group.
    :return: the bundle, with the toy split already made.
    """
    from evd.data.synthetic import NORMAL_IDX, SyntheticSpec, generate
    from evd.utils import env

    names = ("n_normal", "n_anomalous", "num_frames", "size", "fps", "seed")
    fields = {f: cfg[f] for f in names if f in cfg}
    spec = SyntheticSpec(**fields)
    out = Path(cfg["out_dir"]) if cfg.get("out_dir") else env.cache_dir() / "smoke" / "synthetic"
    world = generate(out, spec)

    per_class: dict[int, list[str]] = {}
    normals = set(world.normal_ids)
    for video_id in sorted(v.video_id for v in world.videos):
        annotation = world.annotations[video_id]
        class_name = NORMAL_IDX if video_id in normals else annotation.intervals[0][2]
        per_class.setdefault(int(class_name), []).append(video_id)
    query = tuple(sorted(v for group in per_class.values() for v in group[1::2]))
    train = tuple(sorted(v for group in per_class.values() for v in group[0::2]))

    return DatasetBundle(
        dataset_id=str(cfg.get("id", "synth")),
        name="synthetic",
        classes=world.classes,
        normal_idx=NORMAL_IDX,
        inventory={v.video_id: v for v in world.videos},
        annotations=world.annotations,
        train_videos=train,
        query_videos=query,
        annotation_source=f"synthetic@{world.spec.hash()[:12]}",
    )


# --------------------------------------------------------------------------- #
# UCF-Crime
# --------------------------------------------------------------------------- #

#: The two files selecting the 1900 canonical videos among the 1950 physical ones.
SPLIT_TRAIN = "Anomaly_Train.txt"
SPLIT_TEST = "Anomaly_Test.txt"

#: Allowed annotation adapters, by config name
#: (``configs/data/ucf_crime.yaml::annotations.source``). Only ``ucf_lab`` is wired:
#: Sultani covers only the 290 test videos and is a comparison oracle, not a source.
ANNOTATION_SOURCES = ("ucf_lab",)


def _path(cfg: Mapping[str, Any], chiave: str, what: str) -> Path:
    value = cfg.get(chiave)
    if not value:
        raise ValueError(
            f"configs/data: '{chiave}' is missing ({what}). On the cluster it comes from "
            f"hpc/env.sh; locally it must be passed explicitly"
        )
    return Path(str(value))


@DATASOURCES.register("ucf_crime")
def _ucf_crime(cfg: Mapping[str, Any]) -> DatasetBundle:
    """UCF-Crime with the laboratory annotations (ADR-022).

    * The 1900 canonical videos are selected by the official splits, not the inventory
      (the 50 files of ``z_Normal_Videos_event`` are out-of-split copies).
    * Clamping annotations needs the inventory ``num_frames``.
    * The class comes from the interval, never from the folder (216 of 1831 intervals
      disagree, ADR-022); the adapter does this.

    :param cfg: the resolved ``configs/data/ucf_crime.yaml`` group.
    :return: the bundle with the 1900 canonical videos and loaded annotations.
    :raises ValueError: if a path is missing or the annotation source is not allowed.
    :raises FileNotFoundError: if inventory, splits or annotations do not exist.
    """
    root = _path(cfg, "root", "the UCF_Crimes root")
    inventory_csv = Path(str(cfg.get("inventory") or root / "inventory.csv"))
    splits_dir = Path(str(cfg.get("splits_dir") or root / "Anomaly_Detection_splits"))
    annotations = cfg.get("annotations") or {}
    source = str(annotations.get("source", ""))
    if source not in ANNOTATION_SOURCES:
        raise ValueError(
            f"data.annotations.source = {source!r} is not among {ANNOTATION_SOURCES}. "
            f"The config holds '???' until the choice is declared (ADR-022)"
        )

    record = read_records(inventory_csv)
    canonical = load_split_paths([splits_dir / SPLIT_TRAIN, splits_dir / SPLIT_TEST])
    per_path = {r.relative_path: r for r in record}
    missing = sorted(p for p in canonical if p not in per_path)
    if missing:
        raise ValueError(
            f"{len(missing)} split paths are not in the inventory "
            f"({missing[:3]}...): do inventory and splits come from different releases?"
        )

    inventory = {
        r.video_id: VideoMeta(
            video_id=r.video_id,
            path=str(root / r.relative_path),
            num_frames=r.num_frames,
            fps=r.fps,
            width=r.width,
            height=r.height,
            codec=r.codec,
        )
        for r in per_path.values()
    }
    num_frames = {vid: meta.num_frames for vid, meta in inventory.items()}

    folder = _path(annotations, "path", "the annotation folder")
    adapter = UcfLabAnnotations.from_dir(folder, num_frames=num_frames)
    loaded = adapter.load()
    clamp = adapter.clamps()
    if clamp:
        # Counted and declared, not hidden (Q20, ADR-022).
        log.info("annotations: %d intervals clamped to num_frames - 1", len(clamp))

    train = tuple(
        sorted(per_path[p].video_id for p in load_split_paths([splits_dir / SPLIT_TRAIN]))
    )
    query = tuple(sorted(per_path[p].video_id for p in load_split_paths([splits_dir / SPLIT_TEST])))
    classes = ucf_classes()
    return DatasetBundle(
        dataset_id=str(cfg.get("id", "ucf")),
        name="ucf_crime",
        classes=classes,
        normal_idx=normal_index(classes),
        inventory=inventory,
        annotations=loaded,
        train_videos=train,
        query_videos=query,
        annotation_source=f"{source}@{file_sha256(adapter.path)}",
        content_sha={r.video_id: r.sha256 for r in per_path.values()},
    )


# --------------------------------------------------------------------------- #
# MIVIA-Crime, the laboratory's private test set
# --------------------------------------------------------------------------- #

#: Allowed MIVIA-Crime annotations: same file family as FS-UCF-Crime
#: (``evd.annotations.ucf_lab``) under a different name, so ``annotation_source`` states
#: the actual delivery.
MIVIA_ANNOTATION_SOURCES = ("mivia_lab",)


@DATASOURCES.register("mivia_crime")
def _mivia_crime(cfg: Mapping[str, Any]) -> DatasetBundle:
    """MIVIA-Crime: 270 videos, evaluation only (delivered on 10 September 2026).

    Used to measure how a model trained on UCF-Crime holds on another dataset, so
    ``train_videos`` is empty. Facts of this delivery:

    * there is no official split: the evaluation pool is the inventory;
    * video geometry is heterogeneous (47 resolutions from 320×240 to 3840×2160, 6 to 60
      fps); windowing works in seconds with each video's native fps and preprocessing is
      ``squash`` to 384×384, but the inventory is built by decoding because ``nb_frames``
      is missing on some videos;
    * classes are the same 13 as UCF-Crime, so catalogue and ``normal_idx`` do not change.

    A loader item is the window at native resolution: about 800 MB for a 4K window against
    about 7 MB on UCF-Crime, so RAM, not the GPU, limits the number of workers.

    :param cfg: the resolved ``configs/data/mivia_crime.yaml`` group.
    :return: the bundle, with all 270 videos in ``query_videos``.
    :raises ValueError: if a path is missing, the annotation source is not allowed, or two
        videos share a ``video_id``.
    :raises FileNotFoundError: if inventory or annotations do not exist.
    """
    root = _path(cfg, "root", "the MIVIA-Crime test set root")
    inventory_csv = Path(str(cfg.get("inventory") or root / "inventory.csv"))
    annotations = cfg.get("annotations") or {}
    source = str(annotations.get("source", ""))
    if source not in MIVIA_ANNOTATION_SOURCES:
        raise ValueError(
            f"data.annotations.source = {source!r} is not among {MIVIA_ANNOTATION_SOURCES} "
            f"for the mivia_crime dataset"
        )

    record = read_records(inventory_csv)
    if not record:
        raise ValueError(f"{inventory_csv} is empty: the inventory was not built")
    # Bundle keys are `video_id` values: a duplicate would silently drop a video.
    count = Counter(r.video_id for r in record)
    duplicates = sorted(v for v, n in count.items() if n > 1)
    if duplicates:
        raise ValueError(
            f"{inventory_csv}: repeated video_id ({duplicates[:3]}...). "
            f"On MIVIA-Crime the file name is the identity: two homonymous files in two "
            f"class folders are a delivery problem, not a code problem"
        )

    inventory = {
        r.video_id: VideoMeta(
            video_id=r.video_id,
            path=str(root / r.relative_path),
            num_frames=r.num_frames,
            fps=r.fps,
            width=r.width,
            height=r.height,
            codec=r.codec,
        )
        for r in record
    }
    num_frames = {vid: meta.num_frames for vid, meta in inventory.items()}

    folder = Path(str(annotations.get("path") or root))
    adapter = UcfLabAnnotations.from_dir(folder, delivery=MIVIA_CRIME, num_frames=num_frames)
    loaded = adapter.load()
    clamp = adapter.clamps()
    if clamp:
        log.info("MIVIA-Crime annotations: %d intervals clamped to num_frames - 1", len(clamp))
    without_row = sorted(set(inventory) - set(loaded))
    if without_row:
        # The delivery declares one row for each of the 270 videos, so a missing row is a
        # misalignment between the two artifacts and raises.
        raise ValueError(
            f"{len(without_row)} inventory videos have no row in "
            f"{adapter.path.name} ({without_row[:3]}...): do inventory and annotations come "
            f"from two different deliveries?"
        )

    classes = ucf_classes()
    return DatasetBundle(
        dataset_id=str(cfg.get("id", "mivia")),
        name="mivia_crime",
        classes=classes,
        normal_idx=normal_index(classes),
        inventory=inventory,
        annotations=loaded,
        train_videos=(),
        query_videos=tuple(sorted(inventory)),
        annotation_source=f"{source}@{file_sha256(adapter.path)}",
        content_sha={r.video_id: r.sha256 for r in record},
    )


def videos_of(bundle: DatasetBundle, video_ids: Sequence[str]) -> list[VideoMeta]:
    """:return: metadata of the requested videos, in the given order.
    :raises KeyError: at the first missing video, naming it.
    """
    return [bundle.meta(v) for v in video_ids]


#: Keys of the ``data`` group a command line may override: overrides say where files
#: are, never what is read (invariant 1). Shared here instead of being copied in
#: several scripts.
DATA_PATH_OVERRIDES = frozenset(
    {"root", "videos_dir", "splits_dir", "inventory", "annotations.path", "feature_store.root"}
)


def apply_overrides(gruppo: Any, overrides: Sequence[str]) -> None:
    """Apply ``key=value`` overrides to the ``data`` group, in place.

    :param gruppo: the ``data`` group as a ``DictConfig``; modified in place.
    :param overrides: ``key=value`` strings from the command line.
    :raises ValueError: if a string is not of the form ``key=value`` or the key is not in
        :data:`DATA_PATH_OVERRIDES`.
    """
    from omegaconf import OmegaConf

    for override in overrides:
        key, _, value = override.partition("=")
        if not _:
            raise ValueError(f"--set expects 'key=value', not {override!r}")
        key = key.strip()
        if key not in DATA_PATH_OVERRIDES:
            raise ValueError(
                f"--set {key!r} is not allowed: overrides say WHERE files are, "
                f"not what is read. Allowed: {sorted(DATA_PATH_OVERRIDES)}"
            )
        OmegaConf.update(gruppo, key, value, merge=False)


#: Variables resolved by the ``configs/data/*.yaml`` groups with ``oc.env``. OmegaConf
#: reads ``os.environ``, not the ``hpc/env.sh`` defaults, so missing ones are filled with
#: ``setdefault`` (a missing ``oc.env`` fails with an unhelpful error).
DATA_VARIABLES = (
    "EVD_MIVIA_DIR",
    "EVD_DATA_DIR",
    "EVD_UCF_DIR",
    "EVD_ANNOT_DIR",
    "EVD_FEATURES_DIR",
)


def load_data_group(
    nome: str, overrides: Sequence[str] = (), *, configs_dir: Path | None = None
) -> dict[str, Any]:
    """Load ``configs/data/{nome}.yaml`` and apply the path overrides.

    Shared by ``valuta_esterno.py`` and ``estrai_descrittori.py``, including the
    ``setdefault`` on :data:`DATA_VARIABLES`.

    :param nome: group name, without extension.
    :param overrides: ``key=value``, limited to :data:`DATA_PATH_OVERRIDES`.
    :param configs_dir: groups folder; ``None`` means the repository ``configs/data``.
    :return: the resolved group, as a dict.
    :raises FileNotFoundError: if the group does not exist, naming the available ones.
    """
    import os

    from omegaconf import OmegaConf

    from evd.utils import env

    folder = Path(configs_dir) if configs_dir is not None else env.repo_root() / "configs" / "data"
    path = folder / f"{nome}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in folder.glob("*.yaml"))
        raise FileNotFoundError(f"{path} does not exist: available groups are {available}")
    for key in DATA_VARIABLES:
        try:
            os.environ.setdefault(key, str(env.get(key)))
        except Exception:  # noqa: BLE001 - a variable without default is not an error here
            pass
    group = OmegaConf.load(path)
    apply_overrides(group, overrides)
    return OmegaConf.to_container(group, resolve=True)  # type: ignore[return-value]
