"""A declared grid of runs, and how to execute it unattended.

The point is knowing what is missing: a campaign spans days and several wall-time
slices and is relaunched many times. Three decisions:

1. Deduplication asks the registry, not the folder: ``runs/{run_id}/`` exists as soon
   as the first checkpoint is written, only the registry says ``status == "ok"``.
2. The SLURM array index is the position in the whole plan, not in the missing list:
   a partial relaunch is a sparse array (``-a 0,3,7``) and a cell index never changes.
3. A cell ``run_id`` is computed by :func:`evd.exp.runner.run_id_for`, the same function
   that writes it.

A plan can be printed before the protocol freeze: composing a config does not resolve
its ``???``; resolution happens in the ``validate`` command of ``scripts/campagna.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evd.exp.registry import RegistryRow, completed
from evd.exp.runner import run_id_for
from evd.utils.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "Cell",
    "SweepSpec",
    "submit_commands",
    "fette",
    "plan_missing",
    "piano",
]

#: Account cap: 10 queued jobs, and every array task counts (``hpc/README.md`` §1);
#: more are rejected with ``AssocMaxSubmitJobLimit``.
ACCOUNT_CAP = 10

#: Cells per submission when the sweep does not say otherwise; eight leaves room for a
#: smoke test or a diagnostic run.
MAX_SLICE = 8

#: Sweep fields that are not grid axes.
_NON_AXES = frozenset(
    {"experiment", "assi", "official", "paralleli", "fetta_massima", "risorse", "note"}
)


@dataclass(frozen=True)
class SweepSpec:
    """A grid declared in ``configs/sweep/``.

    ``configs/sweep/`` is not a Hydra group on purpose: it would end up in the run
    ``config.yaml``, while a run does not know it belongs to a grid. This file describes
    an execution, not an experiment.

    :param nome: file name, identifying the campaign in logs.
    :param experiment: file of ``configs/experiment/`` each cell starts from (ADR-005/014).
    :param assi: override name to values; the name is the Hydra key (``adapt``, ``seed``,
        ``train.max_steps``...), so a new axis needs no new code.
    :param official: what the campaign is expected to produce; checked against the
        ``run_id`` grammar.
    :param paralleli: the ``%N`` of the array (concurrent tasks).
    :param fetta_massima: cells per submission.
    :param risorse: options for ``hpc/invia.sh`` (``p``, ``t``, ``c``, ``g``).
    :param note: free text printed with the plan.
    """

    nome: str
    experiment: str
    assi: Mapping[str, tuple[Any, ...]]
    official: bool | None = None
    paralleli: int = 4
    fetta_massima: int = MAX_SLICE
    risorse: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.assi:
            raise ValueError(
                f"sweep {self.nome!r} without axes: an empty grid is not a campaign. "
                f"Declare at least `assi: {{seed: [1, 2]}}`"
            )
        for axis, values in self.assi.items():
            if not values:
                raise ValueError(f"axis {axis!r} of {self.nome!r} has no values")
        if self.paralleli <= 0:
            raise ValueError(f"paralleli must be > 0, not {self.paralleli}")
        if not 0 < self.fetta_massima <= ACCOUNT_CAP:
            raise ValueError(
                f"fetta_massima must be between 1 and {ACCOUNT_CAP}: the account cap is "
                f"{ACCOUNT_CAP} queued jobs and every array task counts (hpc/README.md). "
                f"The default is {MAX_SLICE}, which leaves room for a diagnostic run"
            )

    @property
    def n_celle(self) -> int:
        n = 1
        for values in self.assi.values():
            n *= len(values)
        return n

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepSpec:
        """Read a sweep from ``configs/sweep/*.yaml``.

        :param path: the file.
        :return: the validated spec.
        :raises ValueError: if ``experiment`` or ``assi`` is missing, or an unknown field is
            present.
        """
        import yaml

        path_value = Path(path)
        data = yaml.safe_load(path_value.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path_value}: the file is not a YAML mapping")
        unknown = sorted(set(data) - _NON_AXES)
        if unknown:
            raise ValueError(
                f"{path_value}: unrecognized fields {unknown}. Axes go under `assi:`, "
                f"not at the root, otherwise nobody would read them"
            )
        if not data.get("experiment"):
            raise ValueError(
                f"{path_value}: `experiment` is missing. A campaign always starts from a file of "
                f"configs/experiment/, never from loose overrides (ADR-014)"
            )
        axes = data.get("assi") or {}
        return cls(
            nome=path_value.stem,
            experiment=str(data["experiment"]),
            assi={str(k): tuple(v) for k, v in axes.items()},
            official=data.get("official"),
            paralleli=int(data.get("paralleli", 4)),
            fetta_massima=int(data.get("fetta_massima", MAX_SLICE)),
            risorse={str(k): str(v) for k, v in (data.get("risorse") or {}).items()},
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True)
class Cell:
    """A grid cell: one run, with the index addressing it in an array.

    :param index: position in the whole plan, stable across relaunches.
    :param run_id: the run identity, computed by :func:`evd.exp.runner.run_id_for`.
    :param overrides: Hydra overrides composing this cell.
    :param valori: axis to value, for printing and the plan JSON.
    :param official: whether the ``run_id`` makes it an official run.
    """

    index: int
    run_id: str
    overrides: tuple[str, ...]
    valori: Mapping[str, Any]
    official: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "run_id": self.run_id,
            "overrides": list(self.overrides),
            "valori": dict(self.valori),
            "official": self.official,
        }


def _overrides(spec: SweepSpec, combination: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the Hydra overrides of a cell, in stable order.

    ``+experiment=`` comes first because it fixes the groups; axes override it.
    """
    return (
        f"+experiment={spec.experiment}",
        *(f"{axis}={value}" for axis, value in combination.items()),
    )


def piano(spec: SweepSpec, *, config_dir: str | Path) -> list[Cell]:
    """Expand the grid into cells, composing each config with Hydra.

    Composing guarantees that the plan ``run_id`` is the one the run will write. Composition
    does not resolve interpolations or ``???``.

    :param spec: the grid.
    :param config_dir: ``configs/`` of the repository.
    :return: cells, in Cartesian-product order of the axes.
    :raises ValueError: if a cell produces a ``run_id`` outside the grammar, or declared and
        deduced ``official`` disagree.
    """
    from hydra import compose, initialize_config_dir

    axes = dict(spec.assi)
    combinations = [
        dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())
    ]
    cells: list[Cell] = []
    with initialize_config_dir(config_dir=str(Path(config_dir).resolve()), version_base="1.3"):
        for index, combination in enumerate(combinations):
            overrides = _overrides(spec, combination)
            cfg = compose(config_name="config", overrides=list(overrides))
            identity = run_id_for(
                {
                    "exp_id": _field(cfg, "exp_id", overrides),
                    "backbone": {"id": _field(cfg, "backbone.id", overrides)},
                    "adapt": {"name": _field(cfg, "adapt.name", overrides)},
                    # No `k`: `run_id_for` no longer reads it (ADR-031).
                    "seed": _field(cfg, "seed", overrides),
                },
                dataset_id=str(_field(cfg, "data.id", overrides)),
            )
            cells.append(
                Cell(
                    index=index,
                    run_id=identity.format(),
                    overrides=overrides,
                    valori=combination,
                    official=identity.is_official,
                )
            )
    _check_official(spec, cells)
    _check_uniqueness(spec, cells)
    return cells


def _field(cfg: Any, percorso: str, overrides: Sequence[str]) -> Any:
    """Return a field of the composed config, with an error saying what is missing.

    ``OmegaConf.select`` instead of direct access, so the error names the cell; a missing
    key and ``???`` both give ``None``.

    :raises ValueError: if the field is missing or still ``???``.
    """
    from omegaconf import OmegaConf

    value = OmegaConf.select(cfg, percorso, default=None)
    if value is None:
        raise ValueError(
            f"cell {list(overrides)} does not declare {percorso!r}: without it the run_id does "
            f"not exist and the campaign cannot know whether this cell was executed. "
            f"Add an axis fixing it, or write it in configs/experiment/"
        )
    return value


def _check_official(spec: SweepSpec, celle: Sequence[Cell]) -> None:
    """Check that what the sweep declares matches what the grammar decides.

    :raises ValueError: if they differ (a trial sweep believing itself official would write
        ``official: true`` registry rows).
    """
    if spec.official is None:
        return
    discordant = [c.run_id for c in celle if c.official is not spec.official]
    if discordant:
        raise ValueError(
            f"sweep {spec.nome!r} declares official={spec.official} but {len(discordant)} "
            f"cells are {'not ' if spec.official else ''}official by grammar "
            f"(e.g. {discordant[0]}). A run is official if `exp` is not 'smoke' "
            f"(docs/05 §8): change `exp_id` or the declaration"
        )


def _check_uniqueness(spec: SweepSpec, celle: Sequence[Cell]) -> None:
    """Check that no two cells share a ``run_id`` (two writers on the same run).

    :raises ValueError: naming the duplicated ``run_id``; it happens when a grid axis does
        not enter the ``run_id`` (e.g. ``train.max_steps``).
    """
    seen: dict[str, int] = {}
    for cell in celle:
        if cell.run_id in seen:
            raise ValueError(
                f"sweep {spec.nome!r} generates {cell.run_id} twice: cells "
                f"{seen[cell.run_id]} and {cell.index} differ by an axis that does **not** "
                f"enter the run_id, so in the registry they would be the same run and one "
                f"checkpoint would resume the other"
            )
        seen[cell.run_id] = cell.index


def plan_missing(celle: Iterable[Cell], righe: Iterable[RegistryRow]) -> list[Cell]:
    """Return the cells the registry does not report as complete.

    :param celle: the plan.
    :param righe: lines from :func:`evd.exp.registry.read_registry`.
    :return: cells to execute, in plan order.
    """
    done = completed(righe)
    return [c for c in celle if c.run_id not in done]


def fette(indices: Sequence[int], *, maximum: int = MAX_SLICE) -> list[list[int]]:
    """Split the indices into submissions below the account cap.

    :param indici: plan positions to execute.
    :param massimo: indices per slice.
    :return: the slices, in order.
    :raises ValueError: if ``massimo`` is not positive.
    """
    if maximum <= 0:
        raise ValueError(f"massimo must be > 0, not {maximum}")
    sorted_items = sorted({int(i) for i in indices})
    return [sorted_items[i : i + maximum] for i in range(0, len(sorted_items), maximum)]


def spec_array(indices: Sequence[int], *, paralleli: int) -> str:
    """Return the ``--array`` string of a slice.

    Sparse rather than a range: indices are positions in the whole plan, so a partial
    relaunch is ``0,3,7``, not ``0-2``.

    :return: e.g. ``"0,3,7%4"``.
    :raises ValueError: on an empty slice.
    """
    if not indices:
        raise ValueError("empty slice: nothing to submit")
    return ",".join(str(i) for i in indices) + f"%{paralleli}"


def submit_commands(
    spec: SweepSpec,
    celle: Sequence[Cell],
    *,
    submit_sh: str | Path,
    sbatch: str | Path,
    job_name: str | None = None,
) -> list[list[str]]:
    """Return the ``hpc/invia.sh`` commands that would execute these cells.

    Always through ``invia.sh``, never plain ``sbatch``: it checks what SLURM reports badly
    or late (missing account, time over the partition limit, GPUs on ``fatq``, array index
    over ``MaxArraySize``).

    :param spec: the grid, providing resources and parallelism.
    :param celle: cells to execute, usually the missing ones.
    :param submit_sh: path of ``hpc/invia.sh``.
    :param sbatch: launcher path, relative to the workspace root.
    :param job_name: ``-J``; defaults to ``evd-<sweep name>``.
    :return: one command per slice, as an argument list (paths may contain spaces).
    """
    commands: list[list[str]] = []
    for slice_ in fette([c.index for c in celle], maximum=spec.fetta_massima):
        command = [str(submit_sh)]
        for option, value in sorted(spec.risorse.items()):
            command += [f"-{option}", str(value)]
        command += [
            "-a",
            spec_array(slice_, paralleli=spec.paralleli),
            "-J",
            job_name or f"evd-{spec.nome}",
            str(sbatch),
            spec.nome,
        ]
        commands.append(command)
    return commands
