#!/usr/bin/env python
"""Experiment entry point: a thin CLI (docs/02 §5).

    python scripts/run_experiment.py +experiment=smoke
    python scripts/run_experiment.py +experiment=smoke --cfg job     # print the resolved config only
    python scripts/run_experiment.py +experiment=smoke seed=2 train.max_steps=5

No logic here: Hydra composes the config and this file passes it to
``ExperimentRunner`` (docs/03 §8). The smoke test is a special case only because it must
first generate the synthetic world and its toy split; it then goes through the same
engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Lets the script run without `pip install -e .` (on the cluster PYTHONPATH is set by
# hpc/env.sh).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evd.utils.logging import get_logger, setup_logging  # noqa: E402

log = get_logger("run_experiment")


# version_base="1.3", not None: with None Hydra emulates 1.1 semantics, where
# `hydra.job.chdir` defaults to True and every relative project path would point
# elsewhere. It is also pinned in configs/config.yaml.
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging()
    exp = cfg.get("exp_id")
    log.info("experiment '%s' (seed=%s)", exp, cfg.get("seed"))

    if exp == "smoke":
        from evd.exp.smoke import SmokeConfig, run_smoke

        out = run_smoke(
            SmokeConfig(
                seed=int(cfg.data.seed),
                run_seed=int(cfg.seed),
                max_steps=int(cfg.train.max_steps),
                batch_size=int(cfg.train.batch_size),
                lr=float(cfg.train.lr_grid[0]),
                eval_every=int(cfg.train.eval_every),
                train_sampling=str(cfg.protocol.temporal.train_sampling),
                hop_s=float(cfg.protocol.temporal.hop_s),
                frame_mapping=str(cfg.protocol.temporal.frame_mapping),
                threshold_quantile=float(cfg.protocol.threshold.quantile),
                device=cfg.get("device"),
                adapt=str(cfg.get("smoke", {}).get("adapt", "none")),
                make_figure=bool(cfg.get("smoke", {}).get("make_figure", True)),
                write_run=bool(cfg.get("smoke", {}).get("write_run", True)),
                tiny={
                    "dim": int(cfg.backbone.dim),
                    "depth": int(cfg.backbone.depth),
                    "num_heads": int(cfg.backbone.num_heads),
                    "size": int(cfg.backbone.size),
                    "num_frames": int(cfg.backbone.num_frames),
                    "rope": bool(cfg.backbone.rope),
                },
                world={
                    "n_normal": int(cfg.data.n_normal),
                    "n_anomalous": int(cfg.data.n_anomalous),
                    "num_frames": int(cfg.data.num_frames),
                    "size": int(cfg.data.size),
                    "fps": float(cfg.data.fps),
                },
            )
        )
        print(OmegaConf.to_yaml(OmegaConf.create({"result": out})))
        return

    from evd.exp.runner import ExperimentRunner

    result = ExperimentRunner().run(OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]
    print(OmegaConf.to_yaml(OmegaConf.create({"result": result.metrics})))


if __name__ == "__main__":
    main()
