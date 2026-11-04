"""Aggregation over seeds: from many runs to one row with its spread.

Three rules: ``n`` is always reported next to mean and spread; with a single seed the
standard deviation is ``None``, not ``0``; an incomplete group is declared, never
silently aggregated. The module does not know the registry: the bridge is
:func:`evd.exp.registry.grouped_for_aggregation`.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evd.utils.io import ensure_dir
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "GroupKey",
    "SeedAggregator",
    "SeedStat",
    "tidy_rows",
]


@dataclass(frozen=True, order=True)
class GroupKey:
    """What distinguishes one curve from another: every run-id field except the seed.

    ``k`` stays a string, since ``"full"`` and ``"na"`` are legitimate grid values.
    """

    exp: str
    dataset: str
    backbone: str
    adapt: str
    k: str | None = None
    """Only for few-shot regime runs, where ``k`` was part of the group identity;
    ``None`` for current runs."""

    def as_dict(self) -> dict[str, str]:
        """:return: the key as a dictionary; ``k`` appears only for few-shot groups."""
        base = {
            "exp": self.exp,
            "dataset": self.dataset,
            "backbone": self.backbone,
            "adapt": self.adapt,
        }
        if self.k is not None:
            base["k"] = self.k
        return base


@dataclass(frozen=True)
class SeedStat:
    """A metric aggregated over seeds.

    :param metric: metric name, as in ``metrics.json``.
    :param mean: mean over the available seeds.
    :param std: sample standard deviation, or ``None`` with a single seed.
    :param n: number of contributing seeds.
    :param n_expected: number of seeds expected by the schedule, if declared.
    :param seeds: seeds actually present, sorted.
    :param values: values in the same order as ``seeds``.
    """

    metric: str
    mean: float
    std: float | None
    n: int
    seeds: tuple[int, ...] = ()
    values: tuple[float, ...] = ()
    n_expected: int | None = None

    @property
    def complete(self) -> bool:
        """:return: ``False`` if the schedule expected more seeds than were seen."""
        return self.n_expected is None or self.n >= self.n_expected

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "n_expected": self.n_expected,
            "complete": self.complete,
            "seeds": list(self.seeds),
        }


def _number(valore: Any) -> float | None:
    """:return: the value as float if it is a finite number, otherwise ``None``
    (``nan`` values are counted as missing instead of propagating).
    """
    if isinstance(valore, bool) or not isinstance(valore, (int, float)):
        return None
    number = float(valore)
    return number if math.isfinite(number) else None


class SeedAggregator:
    """Collect the runs of a grid and reduce them to mean and spread.

    :param metrics: metrics to aggregate; ``None`` means every scalar metric found.
        Declaring them makes missing metrics visible.
    :param seeds_expected: number of seeds of the schedule, used only to mark
        incomplete groups.
    """

    def __init__(
        self,
        metrics: Sequence[str] | None = None,
        *,
        seeds_expected: int | None = None,
    ) -> None:
        self.metrics = tuple(metrics) if metrics else None
        self.seeds_expected = seeds_expected
        self._values: dict[GroupKey, dict[str, dict[int, float]]] = {}

    def add(self, key: GroupKey, seed: int, metrics: Mapping[str, Any]) -> None:
        """Add one run; the same ``(key, seed)`` seen twice keeps the last one.

        :param key: group of the run.
        :param seed: seed of the run (the aggregation axis).
        :param metrics: metrics of the run.
        """
        per_metric = self._values.setdefault(key, {})
        names = self.metrics if self.metrics is not None else tuple(metrics)
        for name in names:
            number = _number(metrics.get(name))
            if number is None:
                continue
            per_metric.setdefault(name, {})[seed] = number

    def add_all(self, righe: Iterable[tuple[GroupKey, int, Mapping[str, Any]]]) -> None:
        """Add several runs at once.

        :param righe: ``(group, seed, metrics)`` triples.
        """
        for key, seed, metrics in righe:
            self.add(key, seed, metrics)

    def groups(self) -> dict[GroupKey, dict[str, SeedStat]]:
        """:return: the aggregated metrics of every group."""
        outside: dict[GroupKey, dict[str, SeedStat]] = {}
        for key in sorted(self._values):
            per_metric = self._values[key]
            stats: dict[str, SeedStat] = {}
            for name in sorted(per_metric):
                seeds = sorted(per_metric[name])
                values = tuple(per_metric[name][s] for s in seeds)
                stats[name] = SeedStat(
                    metric=name,
                    mean=statistics.fmean(values),
                    # A single seed does not measure a spread: `None`, never 0.
                    std=statistics.stdev(values) if len(values) > 1 else None,
                    n=len(values),
                    seeds=tuple(seeds),
                    values=values,
                    n_expected=self.seeds_expected,
                )
            outside[key] = stats
        return outside

    def incomplete(self) -> list[tuple[GroupKey, str, int]]:
        """:return: ``(group, metric, n)`` for every cell below the seed schedule."""
        return [
            (key, name, stat.n)
            for key, statistics in self.groups().items()
            for name, stat in statistics.items()
            if not stat.complete
        ]

    def tidy(self) -> list[dict[str, Any]]:
        """:return: one row per ``(group, metric)``, the tidy format consumed by
        ``make_figures.py``.
        """
        rows: list[dict[str, Any]] = []
        for key, group_stats in self.groups().items():
            for name in sorted(group_stats):
                stat = group_stats[name]
                rows.append({**key.as_dict(), **stat.as_dict()})
        return rows

    def to_csv(self, path: str | Path) -> Path:
        """Write the tidy CSV.

        :param path: destination; missing folders are created.
        :return: the written path.
        :raises ValueError: if there is nothing to write.
        """
        rows = self.tidy()
        if not rows:
            raise ValueError(
                "no run to aggregate: the registry has no run matching the requested filters"
            )
        destination = Path(path)
        ensure_dir(destination.parent)
        columns = list(rows[0])
        with open(destination, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: _csv(row[c]) for c in columns})
        missing = self.incomplete()
        if missing:
            log.warning(
                "%d cells below the seed schedule: the first is %s/%s with n=%d",
                len(missing),
                missing[0][0].adapt,
                missing[0][1],
                missing[0][2],
            )
        return destination


def _csv(valore: Any) -> Any:
    """Format a value for the CSV: ``None`` stays empty, lists become ``;``-separated text."""
    if valore is None:
        return ""
    if isinstance(valore, (list, tuple)):
        return ";".join(str(v) for v in valore)
    return valore


def tidy_rows(
    righe: Iterable[tuple[GroupKey, int, Mapping[str, Any]]],
    *,
    metrics: Sequence[str] | None = None,
    seeds_expected: int | None = None,
) -> list[dict[str, Any]]:
    """Turn runs into tidy rows in one call.

    :param righe: ``(group, seed, metrics)`` for every run.
    :param metrics: metrics to aggregate.
    :param seeds_expected: seed schedule, to mark incomplete groups.
    :return: the tidy rows.
    """
    aggregator = SeedAggregator(metrics, seeds_expected=seeds_expected)
    aggregator.add_all(righe)
    return aggregator.tidy()
