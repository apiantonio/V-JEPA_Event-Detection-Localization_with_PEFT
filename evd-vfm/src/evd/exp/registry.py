"""Read side of the registry: which runs finished, and what they produced.

``evd.exp.tracker`` writes, this module queries. Writing is strict, reading is tolerant:
the registry holds months of history and a malformed line must not hide what is
missing. Used by ``scripts/campagna.py`` (skip completed runs by asking the registry,
not the folder) and by ``evd.eval.aggregate``.

Format facts:

* one line per execution, not per ``run_id``: the identity key is
  ``(run_id, started_at)`` and the latest line is the current state;
* some historical run ids do not follow the docs/05 §8 grammar; they are read as they
  are and never rewritten.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evd.exp.run_id import RunId
from evd.exp.tracker import REGISTRY_NAME
from evd.utils import env
from evd.utils.io import read_jsonl
from evd.utils.logging import get_logger

if TYPE_CHECKING:  # types only: the real import is inside the bridge function
    from evd.eval.aggregate import GroupKey

log = get_logger(__name__)

__all__ = [
    "RegistryRow",
    "completed",
    "is_complete",
    "latest_per_run",
    "mancanti",
    "grouped_for_aggregation",
    "read_registry",
    "registry_path",
    "rows_for",
]

#: State meaning "this run produced what it had to".
COMPLETE_STATE = "ok"


@dataclass(frozen=True)
class RegistryRow:
    """A registry line, read without requiring it to be perfect.

    :param run_id: identity, not validated (see the module docstring).
    :param status: ``"ok"`` for a finished run; other values are outcomes declared by the
        experiment (``"needs-analysis"``...).
    :param official: false for smoke tests and trials.
    :param started_at: half of the identity key.
    :param finished_at: when the line was written.
    :param git_sha: commit that produced the number.
    :param metrics: dictionary returned by the experiment.
    :param raw: the whole line, for fields not named by this dataclass.
    """

    run_id: str
    status: str = ""
    official: bool = False
    started_at: str = ""
    finished_at: str = ""
    git_sha: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """:return: ``True`` if the run reached the end."""
        return self.status == COMPLETE_STATE

    def parsed_id(self) -> RunId | None:
        """:return: the structured identity, or ``None`` if outside the grammar."""
        try:
            return RunId.parse(self.run_id)
        except ValueError:
            return None

    def get(self, chiave: str, default: Any = None) -> Any:
        """:return: any field of the line, including those not named here."""
        return self.raw.get(chiave, default)


def registry_path(runs_dir: str | Path | None = None) -> Path:
    """:return: where ``registry.jsonl`` lives on this machine (one runs folder per machine,
    mirrored from ``hpc/env.sh`` in ``evd.utils.env.runs_dir()``).
    """
    root = Path(runs_dir) if runs_dir is not None else env.runs_dir()
    return root / REGISTRY_NAME


def read_registry(path: str | Path | None = None) -> list[RegistryRow]:
    """Read the registry without failing on a broken line.

    :param path: the file, or ``None`` for the current machine.
    :return: readable lines in write order; a missing file gives an empty list.
    """
    source = Path(path) if path is not None else registry_path()
    if not source.is_file():
        return []
    rows: list[RegistryRow] = []
    discarded = 0
    for row in read_jsonl(source, skip_invalid=True):
        if not isinstance(row, dict) or not row.get("run_id"):
            discarded += 1
            continue
        rows.append(
            RegistryRow(
                run_id=str(row["run_id"]),
                status=str(row.get("status", "")),
                official=bool(row.get("official", False)),
                started_at=str(row.get("started_at", "")),
                finished_at=str(row.get("finished_at", "")),
                git_sha=row.get("git_sha"),
                metrics=row.get("metrics") or {},
                raw=row,
            )
        )
    if discarded:
        log.warning("%s: %d unreadable lines ignored", source, discarded)
    return rows


def latest_per_run(righe: Iterable[RegistryRow]) -> dict[str, RegistryRow]:
    """Reduce the history to the current state of each ``run_id``.

    Lines are sorted by ``started_at``; ties are broken by write order.

    :param righe: lines read.
    :return: ``run_id`` to most recent line.
    """
    last: dict[str, RegistryRow] = {}
    for row in righe:
        previous = last.get(row.run_id)
        if previous is None or row.started_at >= previous.started_at:
            last[row.run_id] = row
    return last


def completed(
    righe: Iterable[RegistryRow],
    *,
    official: bool | None = None,
    filters: Mapping[str, Any] | None = None,
) -> set[str]:
    """Return the identifiers of the runs that reached the end.

    The commit is not part of the criterion: a documentation-only commit must not
    invalidate a campaign.

    :param righe: lines from :func:`read_registry`.
    :param official: if given, keep only runs with that ``official`` value.
    :param filters: fields the line must have with that value, e.g. ``{"exp": "e2d"}``,
        applied to the whole line.
    :return: the completed ``run_id`` values.
    """
    chosen: set[str] = set()
    for run_id, row in latest_per_run(righe).items():
        if not row.complete:
            continue
        if official is not None and row.official is not official:
            continue
        if filters and any(row.get(key) != value for key, value in filters.items()):
            continue
        chosen.add(run_id)
    return chosen


def is_complete(
    run_id: str,
    righe: Iterable[RegistryRow] | None = None,
    *,
    path: str | Path | None = None,
    filters: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a single run is complete.

    :param run_id: identity to look for.
    :param righe: lines already read; if missing, read from ``path``.
    :param path: registry to read when ``righe`` is ``None``.
    :param filters: as in :func:`completed`.
    :return: ``True`` if a finished line exists for that ``run_id``.
    """
    source = list(righe) if righe is not None else read_registry(path)
    return run_id in completed(source, filters=filters)


def mancanti(expected: Sequence[str], righe: Iterable[RegistryRow]) -> list[str]:
    """:return: expected ``run_id`` values the registry does not report as complete, in order."""
    done = completed(righe)
    return [run_id for run_id in expected if run_id not in done]


def rows_for(righe: Iterable[RegistryRow], run_ids: Iterable[str]) -> Iterator[RegistryRow]:
    """:return: the most recent lines of the requested runs, skipping missing ones."""
    last = latest_per_run(righe)
    for run_id in run_ids:
        row = last.get(run_id)
        if row is not None:
            yield row


def grouped_for_aggregation(
    righe: Iterable[RegistryRow],
    *,
    exp: str | Sequence[str] | None = None,
    official: bool | None = True,
) -> list[tuple[GroupKey, int, Mapping[str, Any]]]:
    """Bridge between the registry and :class:`~evd.eval.aggregate.SeedAggregator`.

    Lives here for the import direction (docs/02 §5): ``eval`` sits below ``exp`` and the
    aggregator receives already extracted keys and metrics. Runs outside the grammar are
    skipped with a warning.

    :param righe: lines from :func:`read_registry`.
    :param exp: if given, keep only those experiments (a string or a sequence, e.g.
        ``("e2d", "e3p", "e3r")`` for E3-X.1); ``exp`` stays a ``GroupKey`` field, so cells
        of different experiments are never merged.
    :param official: select by value: ``True`` (default) keeps official runs, ``False``
        non-official ones, ``None`` does not filter.
    :return: ``(group, seed, metrics)`` for every complete run.
    """
    from evd.eval.aggregate import GroupKey

    # A string is a sequence of strings: without this branch `exp="e3r"` would filter on
    # characters.
    which_exp: frozenset[str] | None
    if exp is None:
        which_exp = None
    elif isinstance(exp, str):
        which_exp = frozenset({exp})
    else:
        which_exp = frozenset(exp)

    outside: list[tuple[GroupKey, int, Mapping[str, Any]]] = []
    skipped = 0
    for _run_id, row in sorted(latest_per_run(righe).items()):
        if not row.complete or (official is not None and row.official is not official):
            continue
        identity = row.parsed_id()
        if identity is None:
            skipped += 1
            continue
        if which_exp is not None and identity.exp not in which_exp:
            continue
        outside.append(
            (
                GroupKey(
                    exp=identity.exp,
                    dataset=identity.dataset,
                    backbone=identity.backbone,
                    adapt=identity.adapt,
                    k=identity.k_legacy,
                ),
                identity.seed,
                row.metrics,
            )
        )
    if skipped:
        log.warning(
            "%d complete runs have a run_id outside the grammar and cannot be grouped",
            skipped,
        )
    return outside
