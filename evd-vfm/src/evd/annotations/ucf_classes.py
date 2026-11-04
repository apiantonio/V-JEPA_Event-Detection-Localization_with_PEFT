"""UCF-Crime classes: alphabetical anomaly classes with ``Normal`` last (index 13)."""

from __future__ import annotations

from evd.annotations.schema import NORMAL_CLASS_NAME
from evd.types import EventClass

UCF_ANOMALY_CLASSES: tuple[str, ...] = (
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
)

#: Folders that contain normal videos. File names differ between the official
#: splits, so the folder is the reliable signal; ``Normal_Videos`` is the
#: MIVIA-Crime folder.
NORMAL_SPLIT_DIRS: frozenset[str] = frozenset(
    {"Testing_Normal_Videos_Anomaly", "Training_Normal_Videos_Anomaly", "Normal_Videos"}
)


def ucf_classes() -> tuple[EventClass, ...]:
    """Return the 13 anomaly classes plus ``Normal``, in index order."""
    return tuple(
        EventClass(idx=i, name=name)
        for i, name in enumerate((*UCF_ANOMALY_CLASSES, NORMAL_CLASS_NAME))
    )


def class_index(classes: tuple[EventClass, ...] | None = None) -> dict[str, int]:
    return {c.name: c.idx for c in (classes or ucf_classes())}


def normal_index(classes: tuple[EventClass, ...] | None = None) -> int:
    return class_index(classes)[NORMAL_CLASS_NAME]


def class_from_relative_path(relative_path: str) -> str:
    """Infer the class from the folder of ``Videos/<Class>/<file>.mp4``.

    :param relative_path: path relative to the dataset root.
    :return: the class name.
    :raises ValueError: if the path has no recognised class folder.
    """
    parts = [p for p in relative_path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"path without a class folder: {relative_path!r}")
    folder = parts[-2]
    if folder in NORMAL_SPLIT_DIRS:
        return NORMAL_CLASS_NAME
    if folder not in UCF_ANOMALY_CLASSES:
        raise ValueError(f"folder not recognised as a UCF-Crime class: {folder!r}")
    return folder
