# shellcheck shell=bash
# hpc/env.sh: environment of this code on the UNISA HPC cluster (lnode.hpc.unisa.it).
#
# Source it, do not execute it:
#     source /shared/tesi_mllm/codice/hpc/env.sh
#
# Every default has the form  : "${NAME:=value}" : a value already in the environment wins,
# so a default changes by exporting the variable before sourcing or submitting. SLURM passes
# the environment of the submission to the job. `evd.utils.env` reads these same lines.
#
# Inputs (weights, videos, trained checkpoints) are read-only and shared. Outputs (runs, logs,
# caches) go to your own folder, $EVD_STORE_ROOT = /shared/tesi_mllm/$USER.

# Functions come before the idempotence guard: exported variables reach child processes and
# jobs, shell functions do not, so a child that exits at the guard would remain without them.
evd_prepare_dirs() {
    mkdir -p \
        "$EVD_LOG_DIR" "$EVD_CACHE_DIR" "$EVD_RUNS_DIR" \
        "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" 2>/dev/null || true
}

# Header of a job log. A cell of a campaign takes its budget from the Hydra config, so the
# profiling variables are marked as not governing it.
evd_summary() {
    _evd_nb=""
    if [ "${EVD_BUDGET_DA_CONFIG:-0}" = "1" ]; then
        _evd_nb="  <- not used: the budget comes from the config"
    fi
    printf '%s\n' "--- EVD environment --------------------------------------------"
    printf '  %-18s %s\n' \
        "EVD_ROOT"        "$EVD_ROOT" \
        "EVD_VJEPA_DIR"   "$EVD_VJEPA_DIR" \
        "EVD_CKPT_DIR"    "$EVD_CKPT_DIR" \
        "EVD_UCF_DIR"     "$EVD_UCF_DIR" \
        "EVD_MIVIA_DIR"   "$EVD_MIVIA_DIR" \
        "EVD_RUNS_DIR"    "$EVD_RUNS_DIR" \
        "EVD_LOG_DIR"     "$EVD_LOG_DIR" \
        "EVD_PYTHON"      "$EVD_PYTHON" \
        "TORCH_DEVICE"    "$TORCH_DEVICE" \
        "EVD_CPUS"        "$EVD_CPUS (omp=$OMP_NUM_THREADS)" \
        "EVD_PRECISION"   "$EVD_PRECISION (tf32=$EVD_TF32)$_evd_nb" \
        "EVD_SEED"        "$EVD_SEED$_evd_nb"
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        printf '  %-18s %s\n' \
            "job"       "${SLURM_JOB_ID} (${SLURM_JOB_NAME:-?})" \
            "partition" "${SLURM_JOB_PARTITION:-?}" \
            "node"      "$(hostname)"
    fi
    printf '%s\n' "----------------------------------------------------------------"
}

# --- login node or compute node? ---------------------------------------------
# `invia.sh` sources this file on the login node (no GPU, one core) and `--export=ALL` carries
# that environment into the job. Values deduced from the allocation are listed in EVD_DEDOTTE
# and deduced again when the context changes; values set by hand are not listed and survive.
_evd_context="${SLURM_JOB_ID:-login}"
if [ -n "${EVD_DEDOTTE:-}" ] && [ "${EVD_CONTESTO:-}" != "$_evd_context" ]; then
    for _evd_v in $EVD_DEDOTTE; do unset "$_evd_v"; done
    unset EVD_DEDOTTE EVD_ENV_CARICATO _evd_v
fi
EVD_CONTESTO="$_evd_context"
export EVD_CONTESTO
unset _evd_context

if [ -n "${EVD_ENV_CARICATO:-}" ]; then
    return 0 2>/dev/null || true
fi

# --- SLURM commands -----------------------------------------------------------
# `module` exists only in interactive shells, so `ssh host sbatch` finds no sbatch.
if ! command -v sbatch >/dev/null 2>&1; then
    for _evd_bin in /cm/shared/apps/slurm/current/bin /cm/shared/apps/slurm/*/bin; do
        if [ -x "$_evd_bin/sbatch" ]; then
            PATH="$PATH:$_evd_bin"
            export PATH
            break
        fi
    done
    unset _evd_bin
fi

# --- where things are ---------------------------------------------------------
EVD_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
: "${EVD_ROOT:=$(cd "$EVD_HPC_DIR/.." && pwd)}"
: "${EVD_VJEPA_DIR:=$EVD_ROOT/vjepa2}"
export EVD_HPC_DIR EVD_ROOT EVD_VJEPA_DIR

: "${EVD_SHARED_ROOT:=/shared/tesi_mllm}"
# Inputs: readable by the group of the project, never written by a job.
: "${EVD_ASSETS_ROOT:=/shared/tesi_mllm/a.apicella97@studenti.unisa.it}"
: "${EVD_DATA_DIR:=$EVD_ROOT/data}"
: "${EVD_RAW_DIR:=$EVD_ASSETS_ROOT/data/raw}"
: "${EVD_CKPT_DIR:=$EVD_ASSETS_ROOT/data/checkpoints}"
: "${EVD_UCF_DIR:=$EVD_RAW_DIR/ucf_crime/UCF_Crimes}"
: "${EVD_ANNOT_DIR:=$EVD_DATA_DIR/annotations}"
: "${EVD_MIVIA_DIR:=$EVD_ROOT/data/mivia_crime}"
if [ -d "$EVD_SHARED_ROOT" ] && [ "$EVD_MIVIA_DIR" = "$EVD_ROOT/data/mivia_crime" ]; then
    EVD_MIVIA_DIR="/mnt/beegfs/mmarseglia/OFFICIAL_MIVIA_TEST_(USED_ALSO_FOR_CDC_CONTEST)"
fi

# Outputs: your own folder, never /home (200 GB for everybody) nor a node's local disk.
if [ -z "${EVD_STORE_ROOT:-}" ] && [ -d "$EVD_SHARED_ROOT" ]; then
    EVD_STORE_ROOT="$EVD_SHARED_ROOT/${USER:-$(id -un)}"
fi
: "${EVD_STORE_ROOT:=$HOME/evd-store}"
: "${EVD_FEATURES_DIR:=$EVD_STORE_ROOT/features}"
: "${EVD_CACHE_DIR:=$EVD_STORE_ROOT/cache}"
: "${EVD_LOG_DIR:=$EVD_STORE_ROOT/log}"
: "${EVD_RUNS_DIR:=$EVD_ROOT/evd-vfm/runs}"
if [ -d "$EVD_SHARED_ROOT" ] && [ "$EVD_RUNS_DIR" = "$EVD_ROOT/evd-vfm/runs" ]; then
    EVD_RUNS_DIR="$EVD_STORE_ROOT/runs"
fi
export EVD_SHARED_ROOT EVD_ASSETS_ROOT EVD_STORE_ROOT EVD_DATA_DIR EVD_RAW_DIR EVD_CKPT_DIR
export EVD_UCF_DIR EVD_MIVIA_DIR EVD_ANNOT_DIR EVD_FEATURES_DIR EVD_CACHE_DIR EVD_LOG_DIR
export EVD_RUNS_DIR

# Third-party caches stay out of $HOME; bytecode stays out of the code folder.
: "${HF_HOME:=$EVD_CACHE_DIR/huggingface}"
: "${TORCH_HOME:=$EVD_CACHE_DIR/torch}"
: "${XDG_CACHE_HOME:=$EVD_CACHE_DIR/xdg}"
: "${PIP_CACHE_DIR:=$EVD_CACHE_DIR/pip}"
: "${MPLCONFIGDIR:=$EVD_CACHE_DIR/matplotlib}"
: "${PYTHONPYCACHEPREFIX:=$EVD_CACHE_DIR/pycache}"
export HF_HOME TORCH_HOME XDG_CACHE_HOME PIP_CACHE_DIR MPLCONFIGDIR PYTHONPYCACHEPREFIX

# --- Python -------------------------------------------------------------------
# The environment that produced the numbers: `hpc/setup/requisiti-hpc.txt` is its `pip freeze`.
: "${EVD_VENV:=$EVD_SHARED_ROOT/venv-hpc}"
if [ -z "${EVD_PYTHON:-}" ]; then
    if [ -x "$EVD_VENV/bin/python" ]; then
        EVD_PYTHON="$EVD_VENV/bin/python"
    else
        EVD_PYTHON="$(command -v python3 || echo python3)"
    fi
fi
export EVD_VENV EVD_PYTHON
export PYTHONUNBUFFERED=1

# `evd` is the code of this repository, `src` is always the Meta source (vjepa2/).
: "${PYTHONPATH:=}"
_evd_prepend() {
    [ -d "$1" ] || return 0
    case ":$PYTHONPATH:" in
        *":$1:"*) return 0 ;;
    esac
    PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"
}
_evd_prepend "$(dirname "$EVD_VJEPA_DIR")"
_evd_prepend "$EVD_VJEPA_DIR"
_evd_prepend "$EVD_ROOT/evd-vfm/src"
unset -f _evd_prepend
export PYTHONPATH

# This folder is a copy of the code without its git repository, so no commit can trace a run:
# `meta.json` records `git.available: false`. Without this, every non-smoke run is rejected.
export EVD_ALLOW_NO_GIT=1

# --- values deduced from the allocation --------------------------------------
EVD_DEDOTTE="${EVD_DEDOTTE:-}"
_evd_deduce() {
    if [ -z "${!1:-}" ]; then
        printf -v "$1" '%s' "$2"
        EVD_DEDOTTE="${EVD_DEDOTTE:+$EVD_DEDOTTE }$1"
    fi
    export "${1?}"
}
_evd_deduce EVD_CPUS "${SLURM_CPUS_PER_TASK:-1}"
# Without these OpenMP opens one thread per core of the NODE (112 or 128), not of the job.
_evd_deduce OMP_NUM_THREADS "$EVD_CPUS"
_evd_deduce MKL_NUM_THREADS "$EVD_CPUS"
_evd_deduce OPENBLAS_NUM_THREADS "$EVD_CPUS"
_evd_deduce NUMEXPR_NUM_THREADS "$EVD_CPUS"
_evd_deduce EVD_NUM_WORKERS "$EVD_CPUS"
# The value counts, not the presence: with --gpus-per-task=0 SLURM sets these variables to -1.
_evd_gpu_ids="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
_evd_gpu_num="${SLURM_GPUS_ON_NODE:-}"
case "$_evd_gpu_ids" in
    ""|"-1"|none|NoDevFiles) _evd_gpu_ids="" ;;
esac
if [ -n "$_evd_gpu_ids" ] || { [ -n "$_evd_gpu_num" ] && [ "$_evd_gpu_num" != "0" ]; }; then
    _evd_deduce TORCH_DEVICE "cuda"
else
    _evd_deduce TORCH_DEVICE "cpu"
fi
unset _evd_gpu_ids _evd_gpu_num
_evd_deduce EVD_SHARD "${SLURM_ARRAY_TASK_ID:-0}"
_evd_deduce EVD_NUM_SHARDS "${SLURM_ARRAY_TASK_COUNT:-1}"
export EVD_DEDOTTE
unset -f _evd_deduce

# --- numeric defaults ---------------------------------------------------------
# The experiment configs use bf16 autocast on FP32 weights and inputs (the path of the Meta
# training code): the budget validation refuses it when this is 0. The other four are the
# defaults of the E0 profiling scripts; training takes budget and seed from the configs.
: "${EVD_ALLOW_BF16:=1}"
: "${EVD_PRECISION:=fp32}"
: "${EVD_TF32:=0}"
: "${EVD_BATCH_SIZE:=1}"
: "${EVD_SEED:=0}"
export EVD_ALLOW_BF16 EVD_PRECISION EVD_TF32 EVD_BATCH_SIZE EVD_SEED

# --- defaults of hpc/invia.sh -------------------------------------------------
# The account is mandatory: the default one of a user has a QOS that accepts no job.
: "${EVD_SLURM_ACCOUNT:=did_tesi_mllm_433}"
: "${EVD_PARTITION:=gpuq}"
: "${EVD_TIME:=02:00:00}"
: "${EVD_CPUS_PER_TASK:=8}"
: "${EVD_GPUS_PER_TASK:=1}"
: "${EVD_MEM_PER_NODE:=}"
export EVD_SLURM_ACCOUNT EVD_PARTITION EVD_TIME EVD_CPUS_PER_TASK EVD_GPUS_PER_TASK EVD_MEM_PER_NODE

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
# Deterministic matmuls in cuBLAS; it must be in the environment before CUDA starts.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

EVD_ENV_CARICATO=1
export EVD_ENV_CARICATO
