#!/usr/bin/env python
"""Run a grid of experiments unattended (F6/D7): a thin CLI.

    python scripts/campagna.py piano   dod_finto
    python scripts/campagna.py stato   dod_finto
    python scripts/campagna.py valida  dod_finto
    python scripts/campagna.py esegui  dod_finto --locale
    python scripts/campagna.py esegui  dod_finto --indice 3      # one cell (SLURM)
    python scripts/campagna.py invia   dod_finto                 # print only
    python scripts/campagna.py invia   dod_finto --conferma      # submit

The five subcommands answer five different questions:

``piano``   what shape does this grid have? Runs before the protocol freeze (composing a
            config does not resolve its ``???``).
``stato``   which numbers already exist? Answered by the registry, not by the folders (a
            ``runs/{run_id}/`` folder exists even for a run that died halfway).
``valida``  is this grid executable? Resolves every cell; a residual ``???`` or an
            unapproved protocol stops here.
``esegui``  runs, locally or inside an array task.
``invia``   builds the ``hpc/invia.sh`` commands; prints without submitting unless
            ``--conferma`` is given.

The grid, deduplication and slices live in :mod:`evd.exp.campagna`, execution in
:class:`evd.exp.runner.ExperimentRunner`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from evd.exp.campagna import (  # noqa: E402
    Cell,
    SweepSpec,
    piano,
    plan_missing,
    submit_commands,
)
from evd.exp.registry import read_registry  # noqa: E402
from evd.utils.io import write_json  # noqa: E402
from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("campagna")

SWEEP_DIR = REPO / "configs" / "sweep"
CONFIG_DIR = REPO / "configs"
#: The launcher, relative to the workspace root (where ``hpc/invia.sh`` is run from).
SBATCH = "evd-vfm/scripts/slurm/campagna.sbatch"
SUBMIT = "hpc/invia.sh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="comando", required=True)

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("sweep", help=f"name of a file in {SWEEP_DIR.relative_to(REPO)}/")
        p.add_argument(
            "--registry", type=Path, help="alternative registry (default: the machine one)"
        )
        return p

    common(sub.add_parser("piano", help="print the grid")).add_argument(
        "--out", type=Path, help="also write the plan as JSON"
    )
    common(sub.add_parser("stato", help="what is missing, according to the registry"))
    common(sub.add_parser("valida", help="resolve every cell and check the splits"))

    run = common(sub.add_parser("esegui", help="run the missing cells"))
    run.add_argument("--locale", action="store_true", help="run here, one cell after the other")
    run.add_argument("--indice", type=int, help="a single cell, by position in the plan")
    run.add_argument("--tutte", action="store_true", help="also redo completed cells")

    submit = common(sub.add_parser("invia", help="build the hpc/invia.sh commands"))
    submit.add_argument("--conferma", action="store_true", help="actually submit")
    return parser


def _spec(nome: str) -> SweepSpec:
    path = SWEEP_DIR / f"{nome}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in SWEEP_DIR.glob("*.yaml"))
        raise SystemExit(f"sweep {nome!r} does not exist. Available: {available}")
    return SweepSpec.from_yaml(path)


def _cells_and_missing(args: argparse.Namespace) -> tuple[SweepSpec, list[Cell], list[Cell]]:
    spec = _spec(args.sweep)
    cells = piano(spec, config_dir=CONFIG_DIR)
    rows = read_registry(args.registry)
    return spec, cells, plan_missing(cells, rows)


def _row(cella: Cell, done: bool) -> str:
    sign = "OK " if done else "   "
    values = " ".join(f"{k}={v}" for k, v in cella.valori.items())
    return f"{sign}{cella.index:>3}  {cella.run_id:<52} {values}"


def cmd_plan(args: argparse.Namespace) -> int:
    spec, cells, missing = _cells_and_missing(args)
    to_do = {c.index for c in missing}
    print(f"campaign {spec.nome!r}: {len(cells)} cells from {spec.experiment}")
    if spec.note:
        print(f"  {spec.note}")
    for axis, values in spec.assi.items():
        print(f"  {axis}: {list(values)}")
    print()
    for cell in cells:
        print(_row(cell, cell.index not in to_do))
    print(f"\n{len(cells) - len(missing)}/{len(cells)} complete according to the registry")
    if args.out:
        write_json(
            args.out,
            {
                "sweep": spec.nome,
                "experiment": spec.experiment,
                "assi": {k: list(v) for k, v in spec.assi.items()},
                "celle": [c.as_dict() for c in cells],
            },
        )
        print(f"plan written to {args.out}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    spec, cells, missing = _cells_and_missing(args)
    print(f"campaign {spec.nome!r}: {len(cells) - len(missing)}/{len(cells)} complete")
    for cell in missing:
        print(_row(cell, False))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Resolve every cell: the step a residual ``???`` does not pass.

    There is no ``protocol.approved`` check here: the config does it, since the ``???`` of an
    unapproved protocol are the gate (invariant 1).
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    spec = _spec(args.sweep)
    cells = piano(spec, config_dir=CONFIG_DIR)
    problems: list[str] = []
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"):
        for cell in cells:
            try:
                resolved = OmegaConf.to_container(
                    compose(config_name="config", overrides=list(cell.overrides)), resolve=True
                )
            except Exception as error:  # noqa: BLE001 - the message is the result
                problems.append(f"{cell.run_id}: {type(error).__name__}: {error}")
                continue
            missing = _missing_split(resolved)
            if missing is not None:
                problems.append(f"{cell.run_id}: missing split ({missing})")

    if problems:
        print(f"{len(problems)}/{len(cells)} cells NOT executable:\n")
        for problem in problems[:10]:
            print(f"  {problem}")
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more")
        return 1
    print(f"all {len(cells)} cells can be resolved and have their split")
    _check_sbatch()
    return 0


def _missing_split(cfg: Any) -> str | None:
    """:return: the split path if it does not exist, ``None`` if it does.

    The split is decided by :func:`evd.exp.runner.split_path_for`, the same rule the run
    applies.
    """
    from evd.exp.runner import split_path_for

    try:
        path = split_path_for(cfg)
    except ValueError as error:
        return str(error)
    return None if path.is_file() else str(path)


def _check_sbatch() -> None:
    """Run ``bash -n`` on the launcher: syntax is checked before queueing."""
    path = WORKSPACE / SBATCH
    if not path.is_file():
        print(f"missing launcher: {path}")
        return
    try:
        # The file is passed on stdin, not as an argument (on Windows the bash launcher splits the
        # command line on spaces, and this workspace path contains two), and as bytes, not text
        # (`text=True` would write CRLF line endings and bash would report a syntax error).
        outcome = subprocess.run(
            ["bash", "-n"],
            input=path.read_bytes(),
            capture_output=True,
            check=False,
        )
    except OSError:
        print(f"bash not available: {SBATCH} not checked")
        return
    detail = "ok" if outcome.returncode == 0 else outcome.stderr.decode("utf-8", "replace").strip()
    print(f"bash -n {SBATCH}: {detail}")


def cmd_run(args: argparse.Namespace) -> int:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from evd.exp.runner import ExperimentRunner, TrainingInterrupted

    _, cells, missing = _cells_and_missing(args)
    if args.indice is not None:
        choices = [c for c in cells if c.index == args.indice]
        if not choices:
            raise SystemExit(f"index {args.indice} outside the plan ({len(cells)} cells)")
        # Even with an explicit index the registry decides: this keeps an array relaunched in an
        # `afterany` chain idempotent.
        if not args.tutte and choices[0].index not in {c.index for c in missing}:
            print(f"{choices[0].run_id}: already complete according to the registry, not redone")
            return 0
    else:
        choices = cells if args.tutte else missing
        if not args.locale:
            raise SystemExit(
                "without --indice, --locale is required: running the whole grid in a single process "
                "is a choice, not a default (on the cluster it is an array, see `invia`)"
            )

    if not choices:
        print("nothing to do: all cells are complete")
        return 0

    runner = ExperimentRunner()
    interrupted: list[str] = []
    with initialize_config_dir(config_dir=str(CONFIG_DIR.resolve()), version_base="1.3"):
        for cell in choices:
            log.info("cell %d/%d — %s", cell.index, len(cells), cell.run_id)
            cfg = compose(config_name="config", overrides=list(cell.overrides))
            try:
                outcome = runner.run(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
            except TrainingInterrupted as stop:
                # Not a failure: a wall-time slice ended. The checkpoint exists, and relaunching the same
                # cell resumes it.
                log.warning("%s", stop)
                interrupted.append(cell.run_id)
                continue
            log.info("%s: chunk_auc=%s", cell.run_id, outcome.metrics.get("chunk_auc"))

    if interrupted:
        print(f"\n{len(interrupted)} cells interrupted, to resume by relaunching them:")
        for run_id in interrupted:
            print(f"  {run_id}")
        return 2
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    spec, cells, missing = _cells_and_missing(args)
    if not missing:
        print("nothing to submit: all cells are complete")
        return 0
    commands = submit_commands(spec, missing, submit_sh=SUBMIT, sbatch=SBATCH)
    print(
        f"{len(missing)}/{len(cells)} cells missing -> {len(commands)} submissions "
        f"of at most {spec.fetta_massima} tasks\n"
    )
    for command in commands:
        print("  " + " ".join(command))
    if not args.conferma:
        print(
            "\nNothing was submitted. Run from the workspace root, "
            "or rerun with --conferma.\n"
            "Before a real campaign: `hpc/invia.sh -n ...` shows the sbatch command "
            "without queueing."
        )
        return 0
    for command in commands:
        outcome = subprocess.run(command, cwd=str(WORKSPACE), check=False)
        if outcome.returncode != 0:
            print(f"sottomissione fallita ({outcome.returncode}): {' '.join(command)}")
            return outcome.returncode
    return 0


COMMANDS = {
    "piano": cmd_plan,
    "stato": cmd_status,
    "valida": cmd_validate,
    "esegui": cmd_run,
    "invia": cmd_submit,
}


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    return COMMANDS[args.comando](args)


if __name__ == "__main__":
    raise SystemExit(main())
