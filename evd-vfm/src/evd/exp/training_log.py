"""Read the SLURM log of a training job: checkpoints, phases and the machine it ran on.

The trainer writes a line at every checkpoint (``checkpoint at step N in .../runs/<run_id>/...``,
``evd.train.checkpoint``), the runner one when training ends and one when the scores of the
evaluated partition are on disk; the ``hpc/env.sh`` preamble names job, partition and node.
Together they are the only source of the wall-clock cost of a run: ``history.jsonl`` has no
times and the registry has one row per execution (``docs/11``, 8 September 2026).

Three facts the readers of this module rely on:

* the log carries the time of day, not the date: times here are seconds from the first
  timestamp of the log, made monotonic (a timestamp earlier than the previous one is a
  midnight);
* the trainer validates at step boundaries **before** saving the checkpoint of that step
  (``evd/train/trainer.py``), so a validation at step ``s`` falls inside the checkpoint
  interval ``(s0, s1]`` that contains ``s``; and the ``selection:`` line is written only when
  the criterion improves, so validations are located by step, never by that line;
* the GPU is read from the partition and checked against the node name: a log whose two
  sources disagree is not attributed to any machine.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: The GPU behind each partition of the university cluster (``hpc/README.md``).
PARTITION_GPU: dict[str, str] = {"aiq": "H100", "gpuq": "A100"}

#: The GPU expected from the node name prefix, the independent check on the partition.
NODE_GPU: dict[str, str] = {"ainode": "H100", "gnode": "A100"}

_TIME = r"^(\d{2}):(\d{2}):(\d{2})\s+INFO\s+"
#: Checkpoint line; the path names the run, so the line is attributed without trusting the
#: log file name (an array index).
RE_CHECKPOINT = re.compile(
    _TIME + r"evd\.train\.checkpoint \| checkpoint (?:al passo|at step) (\d+) in "
    r".*[/\\]runs[/\\]([^/\\]+)[/\\]"
)
#: First line of a cell, written by the runner before building anything.
RE_RUN = re.compile(_TIME + r"evd\.exp\.runner \| run (\S+) · (?:seme|seed) \d+")
#: End of the training loop (also when it was interrupted by the wall time).
RE_TRAINING_END = re.compile(
    _TIME + r"evd\.exp\.runner \| training: (\d+) (?:passi|steps), (?:loss finale|final loss)"
)
#: Scores of the evaluated partition written: the evaluation is over.
RE_SCORES = re.compile(
    _TIME + r"evd\.exp\.runner \| (?:score di|scores of) \d+ (?:finestre|windows) "
    r"(?:scritti in|written to) .*[/\\]runs[/\\]([^/\\]+)[/\\]scores\.jsonl"
)
#: Resumption from a checkpoint.
RE_RESUME = re.compile(
    _TIME + r"evd\.train\.checkpoint \| (?:ripresa dal passo|resuming from step) (\d+)"
)
_ANY_TIME = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\s")
RE_PARTITION = re.compile(r"^\s+partizione\s+(\S+)\s*$")
RE_NODE = re.compile(r"^\s+nodo\s+(\S+)")
RE_JOB = re.compile(r"^\s+job\s+(\d+)\s+\(")

#: An interval between two checkpoints longer than this is a stalled job, not a slow step.
MAX_INTERVAL_S: float = 3600.0


@dataclass(frozen=True)
class StepInterval:
    """Two consecutive checkpoints of the same run in the same log.

    :param step0: step of the first checkpoint.
    :param step1: step of the second one, greater than ``step0``.
    :param seconds: wall-clock seconds between the two lines.
    :param validations: validations falling in ``(step0, step1]``.
    """

    step0: int
    step1: int
    seconds: float
    validations: int

    @property
    def steps(self) -> int:
        """:return: training steps in the interval."""
        return self.step1 - self.step0


@dataclass
class RunSegment:
    """What one log says about one run (a log can run several cells in series).

    :param run_id: the run.
    :param start: time of the runner's first line for this run, if present.
    :param checkpoints: ``(time, step)`` of every checkpoint line, in log order.
    :param resumed_from: the step the training resumed from, ``None`` for a fresh start.
    :param training_end: ``(time, steps done)`` of the end of the training loop.
    :param scores_written: time at which the evaluation scores were on disk.
    """

    run_id: str
    start: float | None = None
    checkpoints: list[tuple[float, int]] = field(default_factory=list)
    resumed_from: int | None = None
    training_end: tuple[float, int] | None = None
    scores_written: float | None = None

    def intervals(self, eval_every: int | None) -> list[StepInterval]:
        """Return the intervals between consecutive checkpoints of this segment.

        :param eval_every: validation period in steps, or ``None`` if the run never validates.
        :return: intervals with a positive step count and a duration in ``(0, MAX_INTERVAL_S)``,
            each with the number of validations it contains.
        """
        out: list[StepInterval] = []
        for (t0, s0), (t1, s1) in zip(self.checkpoints, self.checkpoints[1:], strict=False):
            dt, dp = t1 - t0, s1 - s0
            if dp <= 0 or not 0.0 < dt < MAX_INTERVAL_S:
                continue
            validations = 0 if not eval_every else s1 // eval_every - s0 // eval_every
            out.append(StepInterval(s0, s1, dt, validations))
        return out

    @property
    def evaluation_seconds(self) -> float | None:
        """:return: seconds from the end of training to the evaluation scores on disk, or
        ``None`` if this log did not evaluate the run."""
        if self.training_end is None or self.scores_written is None:
            return None
        return self.scores_written - self.training_end[0]


@dataclass
class TrainingLog:
    """A parsed SLURM log.

    :param name: file name.
    :param job_id: the SLURM job id printed by ``hpc/env.sh`` (``SLURM_JOB_ID``).
    :param partition: SLURM partition.
    :param node: execution node.
    :param gpu: ``"A100"``/``"H100"`` when partition and node agree, otherwise ``None``.
    :param segments: one per run met in the log, in order of appearance.
    :param duration_s: seconds between the first and the last timestamp.
    """

    name: str
    job_id: str | None
    partition: str | None
    node: str | None
    gpu: str | None
    segments: dict[str, RunSegment]
    duration_s: float


def gpu_of(partition: str | None, node: str | None) -> str | None:
    """Return the GPU of a job, checking the partition against the node.

    :param partition: SLURM partition, e.g. ``"aiq"``.
    :param node: node name, e.g. ``"ainode01"``.
    :return: the GPU name, or ``None`` if the partition is unknown or the node prefix names
        another GPU.
    """
    gpu = PARTITION_GPU.get(partition or "")
    expected = next((g for prefix, g in NODE_GPU.items() if (node or "").startswith(prefix)), None)
    if gpu is None or (expected is not None and expected != gpu):
        return None
    return gpu


def parse_log(text: str, name: str = "") -> TrainingLog:
    """Parse one SLURM log of the training campaigns.

    :param text: the content of a ``hpc/log/*.out`` file.
    :param name: the file name, for provenance only.
    :return: the parsed log; a log without timestamps has no segments and ``duration_s`` 0.
    """
    partition = node = job = None
    segments: dict[str, RunSegment] = {}
    current: RunSegment | None = None
    offset = 0.0
    previous: float | None = None
    first: float | None = None

    def clock(h: str, m: str, s: str) -> float:
        nonlocal offset, previous, first
        value = int(h) * 3600.0 + int(m) * 60.0 + int(s) + offset
        if previous is not None and value < previous:
            offset += 86400.0
            value += 86400.0
        previous = value
        if first is None:
            first = value
        return value

    def segment(run_id: str) -> RunSegment:
        if run_id not in segments:
            segments[run_id] = RunSegment(run_id)
        return segments[run_id]

    for row in text.splitlines():
        if partition is None and (found := RE_PARTITION.match(row)):
            partition = found.group(1)
        if node is None and (found := RE_NODE.match(row)):
            node = found.group(1)
        if job is None and (found := RE_JOB.match(row)):
            job = found.group(1)
        stamp = _ANY_TIME.match(row)
        if stamp is None:
            continue
        t = clock(*stamp.groups())
        if found := RE_RUN.match(row):
            current = segment(found.group(4))
            if current.start is None:
                current.start = t
        elif found := RE_CHECKPOINT.match(row):
            current = segment(found.group(5))
            current.checkpoints.append((t, int(found.group(4))))
        elif (found := RE_RESUME.match(row)) and current is not None:
            current.resumed_from = int(found.group(4))
        elif (found := RE_TRAINING_END.match(row)) and current is not None:
            current.training_end = (t, int(found.group(4)))
        elif found := RE_SCORES.match(row):
            segment(found.group(4)).scores_written = t
    duration = 0.0 if first is None or previous is None else previous - first
    return TrainingLog(
        name=name,
        job_id=job,
        partition=partition,
        node=node,
        gpu=gpu_of(partition, node),
        segments=segments,
        duration_s=duration,
    )


def read_logs(folders: Iterable[Path]) -> list[TrainingLog]:
    """Parse every ``*.out`` file of the given folders that contains a training checkpoint.

    :param folders: log folders; missing ones are skipped.
    :return: the parsed logs, sorted by file name; a file present in two folders under the
        same name is read once (the first folder wins).
    """
    seen: set[str] = set()
    logs: list[TrainingLog] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.out")):
            if path.name in seen:
                continue
            seen.add(path.name)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "evd.train.checkpoint" in text:
                logs.append(parse_log(text, path.name))
    return sorted(logs, key=lambda log_: log_.name)


@dataclass(frozen=True)
class StepCost:
    """Steady-state cost of a training step and of a validation on one GPU.

    :param s_per_step: median seconds per step over the intervals without validation.
    :param n_intervals: how many intervals the median is over.
    :param validation_s: median seconds of one validation pass (interval time minus its steps
        at ``s_per_step``), or ``None`` if no interval contained exactly one validation.
    :param n_validations: how many validations the median is over.
    :param runs: the runs the intervals come from.
    """

    s_per_step: float
    n_intervals: int
    validation_s: float | None
    n_validations: int
    runs: tuple[str, ...]


def step_cost(
    logs: Sequence[TrainingLog],
    runs: Mapping[str, int | None],
    gpu: str,
    *,
    min_intervals: int = 5,
) -> StepCost | None:
    """Measure the cost of a step and of a validation for a set of runs on one GPU.

    The runs must share the architecture (the cost of a step depends on the model, not on the
    seed or on the step count); the caller decides which runs these are.

    :param logs: the parsed logs (:func:`read_logs`).
    :param runs: ``{run_id: eval_every}`` of the runs to pool.
    :param gpu: ``"A100"`` or ``"H100"``.
    :param min_intervals: minimum intervals without validation for the median to be a
        steady-state cost rather than start-up.
    :return: the cost, or ``None`` with fewer than ``min_intervals`` intervals.
    """
    plain: list[float] = []
    with_validation: list[tuple[float, int]] = []
    sources: set[str] = set()
    for log_ in logs:
        if log_.gpu != gpu:
            continue
        for run_id, seg in log_.segments.items():
            if run_id not in runs:
                continue
            for interval in seg.intervals(runs[run_id]):
                sources.add(run_id)
                if interval.validations == 0:
                    plain.append(interval.seconds / interval.steps)
                elif interval.validations == 1:
                    with_validation.append((interval.seconds, interval.steps))
    if len(plain) < min_intervals:
        return None
    per_step = statistics.median(plain)
    excess = [seconds - steps * per_step for seconds, steps in with_validation]
    return StepCost(
        s_per_step=float(per_step),
        n_intervals=len(plain),
        validation_s=float(statistics.median(excess)) if excess else None,
        n_validations=len(excess),
        runs=tuple(sorted(sources)),
    )
