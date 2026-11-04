"""Run one stage of the automatic-localization campaign on one shard of a test dataset.

Thin CLI over :func:`evd.exp.localization_auto.run_stage`; the stage registry is
``STAGES``. Inputs: the VideoMeta manifest, the validation gate, the sealed readout stage
(S0) and the frozen reduced plan. No test annotation is read.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from evd.exp.demo import live_engine, open_run  # noqa: E402
from evd.exp.localization_auto import read_json, run_stage  # noqa: E402
from evd.exp.localization_cris import CRISStage  # noqa: E402
from evd.exp.localization_ig import IGStage  # noqa: E402
from evd.exp.localization_memory import MemoryStage  # noqa: E402
from evd.exp.localization_perturb import ExtremalStage, PerturbStage  # noqa: E402
from evd.exp.localization_readouts import ReadoutStage  # noqa: E402
from evd.exp.localization_vtcd import (  # noqa: E402
    ABLATION_MASS30,
    ABLATION_SWEEP,
    ABSOLUTE_MASS70,
    VTCDAblationStage,
    VTCDStage,
)
from evd.protocol.chunks import sampler_for_protocol  # noqa: E402
from evd.utils.hashing import file_sha256  # noqa: E402
from localization_auto_preflight import CHECKPOINT_SHA, RUN  # noqa: E402

PLAN = ROOT / "configs/localization_auto_20261016_reduced.json"
#: Stage factories ``(engine, spec, gate_fit_seconds, args) -> stage`` and the fitted
#: files each one reads (hashed into the stage identity).
STAGES = {
    "readouts": (lambda e, s, g, a: ReadoutStage(e, s, g), ()),
    "ig": (lambda e, s, g, a: IGStage(e, s, g), ()),
    "perturb": (
        lambda e, s, g, a: PerturbStage(e, s, g, read_json(a.fits / "channels.json")),
        ("channels.json",),
    ),
    "extremal": (lambda e, s, g, a: ExtremalStage(e, s, g), ()),
    "memory": (
        lambda e, s, g, a: MemoryStage(e, s, g, a.fits),
        (
            "memory_thresholds.json",
            "memory_fit_seconds.json",
            "pca.npz",
            "binary_references.npz",
            "p19_reference.json",
            "p24_bank.npy",
            "p25_contexts.npz",
            "p26_gaussian.npz",
            "p28_pca.npz",
            "p28_bank.npy",
            "p29.pt",
            "p30.pt",
            "p32.pt",
        ),
    ),
    "vtcd": (
        lambda e, s, g, a: VTCDStage(
            e,
            s,
            g,
            a.fits,
            a.output.parent / "ig" / a.dataset,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1),
        ),
        ("vtcd_models.npz", "vtcd_fit_seconds.json"),
    ),
    "vtcd_ablation": (
        lambda e, s, g, a: VTCDAblationStage(
            e,
            s,
            g,
            a.fits,
            a.output.parent / "ig" / a.dataset,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1),
        ),
        ("vtcd_models.npz", "vtcd_fit_seconds.json"),
    ),
    "vtcd_ablation_mass30": (
        lambda e, s, g, a: VTCDAblationStage(
            e,
            s,
            g,
            a.fits,
            a.output.parent / "ig" / a.dataset,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1),
            variants=ABLATION_MASS30,
            name="vtcd_ablation_mass30",
        ),
        ("vtcd_models.npz", "vtcd_fit_seconds.json"),
    ),
    "vtcd_ablation_sweep": (
        lambda e, s, g, a: VTCDAblationStage(
            e,
            s,
            g,
            a.fits,
            a.output.parent / "ig" / a.dataset,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1),
            variants=ABLATION_SWEEP,
            name="vtcd_ablation_sweep",
        ),
        ("vtcd_models.npz", "vtcd_fit_seconds.json"),
    ),
    "vtcd_ig_absolute": (
        lambda e, s, g, a: VTCDAblationStage(
            e,
            s,
            g,
            a.fits,
            a.output.parent / "ig" / a.dataset,
            max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4")) - 1),
            variants=ABSOLUTE_MASS70,
            name="vtcd_ig_absolute",
            mass_modes={"P07-abs-mass70": "absolute"},
        ),
        ("vtcd_models.npz", "vtcd_fit_seconds.json"),
    ),
    "cris": (
        lambda e, s, g, a: CRISStage(e, s, g, a.fits / "cris"),
        ("cris/cris_top.npz",),
    ),
}

#: Stages that read the sealed outputs of another stage; the seal enters the identity.
UPSTREAM = {
    "vtcd": "ig",
    "vtcd_ablation": "ig",
    "vtcd_ablation_mass30": "ig",
    "vtcd_ablation_sweep": "ig",
    "vtcd_ig_absolute": "ig",
}


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--s0", required=True, type=Path, help="root of the readout stage")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fits", type=Path, help="fitted artifacts of the campaign")
    parser.add_argument("--dataset", choices=("ucf_test", "mivia_test"), required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--plan", type=Path, default=PLAN, help="frozen plan (the 17/10 follow-up has its own)"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="chunks to localize instead of the gate: {dataset: {video_id: {indices}}}",
    )
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("Invalid shard")
    plan = read_json(args.plan)
    gate = read_json(args.preflight / "gate.json")
    if gate["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("Gate fitted on a different checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no silent CPU fallback")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = open_run(Path(os.environ["EVD_RUNS_DIR"]) / RUN)
    if file_sha256(run.checkpoint_path) != CHECKPOINT_SHA:
        raise ValueError("Selected checkpoint identity mismatch")
    engine, spec = live_engine(run, device="cuda", batch_size=1, num_workers=0)
    engine.sampler = sampler_for_protocol(spec, run.protocol, validazione=False)
    factory, fit_files = STAGES[args.stage]
    upstream = None
    if args.stage in UPSTREAM:
        seal = (
            args.output.parent / UPSTREAM[args.stage] / "operations" / f"seal-{args.dataset}.json"
        )
        if read_json(seal).get("status") != "complete":
            raise ValueError(f"{UPSTREAM[args.stage]} must be sealed before {args.stage}")
        upstream = file_sha256(seal)
    stage = factory(engine, spec, gate["fit_seconds"], args)
    if list(stage.methods) != plan["stages"][args.stage]:
        raise ValueError("Stage methods differ from the frozen plan")
    runtime = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": engine.scorer.precision,
        "tf32": False,
        "batch_size": 1,
        "threads": 1,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    run_stage(
        stage,
        rows=read_json(args.manifest),
        windows=lambda meta: engine.finestre(Path(meta.path), meta),
        s0_root=args.s0,
        dataset=args.dataset,
        output=args.output,
        shard=args.shard,
        shards=args.shards,
        source_commit=args.source_commit,
        runtime=runtime,
        extra_identity={
            "plan_sha256": file_sha256(args.plan),
            "gate_sha256": file_sha256(args.preflight / "gate.json"),
            "checkpoint_sha256": CHECKPOINT_SHA,
            "input_manifest_sha256": file_sha256(args.manifest),
            "fits_sha256": {name: file_sha256(args.fits / name) for name in fit_files},
            "upstream_seal_sha256": upstream,
            "selection_sha256": file_sha256(args.selection) if args.selection else None,
        },
        selection=None
        if args.selection is None
        else {
            video: entry["indices"]
            for video, entry in read_json(args.selection)[args.dataset].items()
        },
    )


if __name__ == "__main__":
    main()
