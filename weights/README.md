# Trained model weights

The [models-v1 release](https://github.com/apiantonio/V-JEPA_Event-Detection-Localization_with_PEFT/releases/tag/models-v1)
contains inference checkpoints, weights-manifest.json and SHA256SUMS.
The same assets are mirrored in the private development repository.

Download a `.pt` asset and rename it to `checkpoint.pt` in the matching run's
`checkpoint/` directory. In the public edition that is `evd-vfm/runs/<run_id>/checkpoint/`;
in development it is `runs/<run_id>/checkpoint/`. Verify its SHA-256 against the manifest.
The placeholder README in each public run marks this destination.

Use the accompanying config.yaml and meta.json. Sequential VPT models also need the
LoRA/DoRA source run: set EVD_RUNS_DIR to the installed runs directory. Inference assets
retain both last and validation-selected states because historical sequential priors
use the last adapter. They do not include optimizer/RNG state and cannot resume training.
Backbones must be downloaded separately from Meta under their upstream terms.

Project contributions follow ../LICENSE; upstream model terms continue to apply.

From the repository root, list models with `python weights/download.py`; install one
with `python weights/download.py --model <run_id>`, or all with `--all`. Every download
is verified before replacing checkpoint.pt. For private assets use GitHub CLI with authentication.
