# Running the code on the UNISA HPC cluster

Everything is ready on the cluster, for the members of the group `did_tesi_mllm_433`:

| What | Where | Access |
|---|---|---|
| Code, run artifacts, localization artifacts | `/shared/tesi_mllm/codice` | read |
| Python environment (`requisiti-hpc.txt` is its `pip freeze`) | `/shared/tesi_mllm/venv-hpc` | read |
| V-JEPA 2.1 weights | `/shared/tesi_mllm/a.apicella97@studenti.unisa.it/data/checkpoints` | read |
| UCF-Crime videos | `/shared/tesi_mllm/a.apicella97@studenti.unisa.it/data/raw/ucf_crime/UCF_Crimes` | read |
| MIVIA-Crime videos and annotations | `/mnt/beegfs/mmarseglia/OFFICIAL_MIVIA_TEST_(USED_ALSO_FOR_CDC_CONTEST)` | read |
| Trained checkpoints | `evd-vfm/runs/<run_id>/checkpoint`, links to the folder of the runs | read |
| **Your outputs**: runs, logs, caches | `/shared/tesi_mllm/$USER` | created at the first job |

`hpc/env.sh` sets all of this; export a variable before sourcing it to change a value. Nothing
needs to be installed. `hpc/setup/crea_ambiente.sh` builds a private copy of the environment,
if ever needed.

```bash
cd /shared/tesi_mllm/codice
source hpc/env.sh && evd_summary
```

Run everything below from this folder. Jobs go through `hpc/invia.sh` (`-h` lists the options),
which passes the mandatory account `did_tesi_mllm_433`. The login node is not for computing.
`aiq` has the H100 cards, `gpuq` the A100; `fatq` has no GPU. A job stops at 7 h (`fatq`: 4 h) and
each user can have 10 jobs submitted, where every task of an array counts as one.

## 1. Check the installation (CPU, a few minutes)

```bash
hpc/invia.sh -p fatq -g 0 -c 4 -m 32G -t 00:20:00 hpc/jobs/python.sbatch scripts/smoke.py --strict
```

## 2. Test a trained model

The selected weights of a run, one inference pass, no training. The threshold is transferred from
the run. `--limite 4` is a trial on four videos. MIVIA-Crime is 3,485 windows, about 20 minutes on
an H100.

```bash
RUN=e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42
hpc/invia.sh -p aiq -t 02:00:00 -c 14 -g 1 -m 96G -J test-mivia hpc/jobs/python.sbatch \
    scripts/valuta_esterno.py --run "$PWD/evd-vfm/runs/$RUN" --dati mivia_crime --num-workers 8
```

The result is in `$EVD_RUNS_DIR/$RUN/${RUN}__mivia/` and compares with
`evd-vfm/runs/$RUN/${RUN}__mivia/metrics.json`. With `--dati ucf_crime` the same pass runs on the
290 videos of the FS-UCF-Crime test set.

Runs with a checkpoint link: `e2c_…`, `e2d_…`, `e3p_…` (three), `e3r_…` (two), `e3s_…`, `lcb1_…`,
`lcb2_…`, `lcb3s_…` (two), `lcb3g_…` (two). All at seed 42. The other seeds are in
`$EVD_ASSETS_ROOT/runs/`.

## 3. Train

A sweep of `evd-vfm/configs/sweep/` is a grid of cells. `invia` prints the commands of the cells
missing from your registry, and submits them with `--conferma`:

```bash
cd evd-vfm
"$EVD_PYTHON" scripts/campagna.py piano e3r_dora
"$EVD_PYTHON" scripts/campagna.py invia e3r_dora        # print only
cd .. && <the printed command>
```

A cell longer than 7 h (one epoch of DoRA takes about 10.5 h on an H100) saves a checkpoint before
the wall time and continues with a chain: resubmit the same index with `-d afterany:<job id>`.

Cells that continue or extend another run (`e3m*`, `e3s*`, `e3so*`, `lcb*`) read the checkpoint
of their source from your runs folder. Train the source first, or link it:

```bash
mkdir -p "$EVD_RUNS_DIR"
ln -s "$PWD/evd-vfm/runs/e3r_ucf_vjepa21l_lora-r32-qkvomlp_s42" "$EVD_RUNS_DIR/"
```

## 4. Localization

`evd-vfm/runs/localization-class-routing-20261021/` holds, next to the official table, what is
needed to recompute it: the spatial ground truth (`gt-snapshot/`), the chunk lists (`plan/`), the
dictionaries (`fits/`), the IG maps, SLIC regions and class logits (`passes/`, 256 IG nodes;
`ig-nodes/n32/` and `ig-nodes-attentive/n32/`, the official 32 nodes) and the sealed predictions
of the six official stages (`stages/`).

Evaluating a stage needs no GPU. This recomputes `evaluation/f1-n32-offline-attentive`:

```bash
C=$PWD/evd-vfm/runs/localization-class-routing-20261021
hpc/invia.sh -p fatq -g 0 -c 16 -m 64G -t 03:00:00 -J loc-eval hpc/jobs/python.sbatch \
    scripts/localization_class_ap.py evaluate \
    --predictions "$C/stages/ign-n32-attentive" --snapshot "$C/gt-snapshot" \
    --intervals "$C/plan/real_use/intervals.json" \
    --mode offline --frames 1fps --ap-only --iou-grid fine --gt-modes union --workers 16 \
    --output "$EVD_STORE_ROOT/loc/f1-n32-offline-attentive"
```

For the online metrics add `--mode online --ovad-plan "$C/plan/ovad_rows/<detector>.json"`. Stages:
`ign-n32-campaign` (LoRA+VPT base), `ign-n32-b3s`, `ign-n32-b3g`, `ign-n32-attentive` (baseline),
`cmp-ig32` and `cmp-other` (the comparison methods). `localization_class_table.py` builds the
page from the evaluations. RBDC/RBDR and the B4 classifiers are left out.

The main README lists the steps of the whole pipeline; each script explains itself with `--help`.
