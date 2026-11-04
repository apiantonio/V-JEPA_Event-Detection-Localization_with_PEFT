#!/usr/bin/env python
"""Common CLI of the offline explanations; the historical command stays compatible."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evd.exp.demo import live_engine, open_run  # noqa: E402
from evd.exp.explanation_campaign import (  # noqa: E402
    METHODS,
    analyze_video,
    select_windows,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--video-root", type=Path, default=ROOT.parent / "data/raw")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/explain/campaign.yaml")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--window-id", action="append", default=[])
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--query", type=Path)
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=ROOT.parent / "data/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--blur-kernel", type=int)
    parser.add_argument("--select-development", action="store_true")
    parser.add_argument("--render-only", type=Path)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--extend-campaign", type=Path)
    parser.add_argument("--prepared-references", type=Path)
    parser.add_argument("--reference-manifest", type=Path)
    args = parser.parse_args()
    settings = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    blur_kernel = args.blur_kernel if args.blur_kernel is not None else settings["blur_kernel"]
    if args.render_only:
        from apps.demo.explanation import render_campaign

        render_campaign(args.render_only)
        return
    if args.run is None:
        parser.error("--run is required for inference and selection")
    run = open_run(args.run)
    if args.extend_campaign:
        import torch

        from evd.exp.explanation_advanced import (
            extend_campaign,
            model_identity,
            prepare_previews,
            prepare_references,
        )
        from evd.explain.methods import ADVANCED_METHODS

        args.methods = args.methods or list(ADVANCED_METHODS)
        if args.prepared_references is None or args.reference_manifest is None:
            parser.error(
                "--extend-campaign requires --prepared-references and --reference-manifest"
            )
        if set(args.methods) - set(ADVANCED_METHODS):
            parser.error(
                "--extend-campaign accepts the five new methods; select them with --methods"
            )
        advanced_config = (
            ROOT / "configs/explain/advanced.yaml"
            if args.config == ROOT / "configs/explain/campaign.yaml"
            else args.config
        )
        advanced_settings = OmegaConf.to_container(OmegaConf.load(advanced_config), resolve=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        engine, _ = live_engine(
            run, device=args.device, batch_size=1, backbone_checkpoint=args.backbone_checkpoint
        )
        identity = model_identity(run, engine, args.backbone_checkpoint, advanced_settings)
        manifest = json.loads(args.reference_manifest.read_text(encoding="utf-8"))
        references, receipt = prepare_references(
            engine, manifest, args.prepared_references, identity
        )
        extend_campaign(
            run,
            engine,
            args.extend_campaign,
            args.out,
            references=references,
            reference_receipt=receipt,
            manifest=manifest,
            identity=identity,
            methods=tuple(args.methods),
            max_new_windows=args.max_windows,
        )
        if args.render:
            from apps.demo.explanation import render_campaign
            from apps.demo.explanation_advanced import render_global_motion

            prepare_previews(args.out)
            render_campaign(args.out)
            render_global_motion(args.out)
        return
    args.methods = args.methods or list(METHODS[:6])
    if args.select_development:
        rows = [
            json.loads(line) for line in run.score_path.read_text().splitlines() if line.strip()
        ]
        selection = select_windows(
            [r for r in rows if r["split"] == "val"],
            run.threshold,
            per_category=settings["per_category"],
            seed=settings["seed"],
        )
        selection.update(run_id=run.run_id, split="val")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise FileExistsError(args.out)
        write_json(args.out, selection)
        return
    if args.video is None and args.manifest is None:
        parser.error("Specificare --video oppure --manifest")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        paths = {}
        for row in manifest.get("video", []):
            paths[row["video_id"]] = Path(row["percorso"])
        if any(not path.exists() for path in paths.values()):
            inventory = {p.stem: p for p in args.video_root.rglob("*.mp4")}
            paths = {video_id: inventory.get(video_id, path) for video_id, path in paths.items()}
        if "windows" in manifest:
            inventory = {p.stem: p for p in args.video_root.rglob("*.mp4")}
            missing = sorted({r["video_id"] for r in manifest["windows"]} - inventory.keys())
            if missing:
                raise FileNotFoundError(f"Manifest videos missing in {args.video_root}: {missing}")
            paths = {r["video_id"]: inventory[r["video_id"]] for r in manifest["windows"]}
        split = manifest.get("split", args.split)
    else:
        paths = {args.video.stem: args.video}
        manifest, split = {}, args.split
    query = json.loads(args.query.read_text()) if args.query else None
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing videos: {missing_paths}")
    engine, _ = live_engine(
        run, device=args.device, batch_size=1, backbone_checkpoint=args.backbone_checkpoint
    )
    # One snapshot per process: later workspace edits do not retroactively change the
    # provenance of the windows still to process.
    source_root = ROOT / "src/evd"
    source_snapshot = {
        str(p.relative_to(source_root)): p.read_bytes() for p in source_root.rglob("*.py")
    }
    for video_id, path in paths.items():
        ids = tuple(
            r["window_id"] for r in manifest.get("windows", []) if r["video_id"] == video_id
        ) or tuple(args.window_id)
        folder = args.out / video_id
        if (folder / "campaign.json").exists():
            raise FileExistsError(f"Incomplete or different campaign: {folder}")
        analyze_video(
            run,
            path,
            folder,
            methods=tuple(args.methods),
            backbone_checkpoint=args.backbone_checkpoint,
            device=args.device,
            split=split,
            window_ids=ids,
            max_windows=args.max_windows,
            query=query,
            perturb=args.perturb,
            blur_kernel=blur_kernel,
            video_root=args.video_root,
            engine=engine,
            settings=settings,
            source_snapshot=source_snapshot,
        )
        if args.render:
            from apps.demo.explanation import render_campaign

            render_campaign(folder)


if __name__ == "__main__":
    main()
