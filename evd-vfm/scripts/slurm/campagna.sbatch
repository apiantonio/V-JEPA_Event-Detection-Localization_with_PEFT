#!/bin/bash
# One task = one cell of a campaign declared in evd-vfm/configs/sweep/.
#
# Always through hpc/invia.sh. `campagna.py invia <sweep>` prints the commands of the missing
# cells; their form is:
#
#   hpc/invia.sh -p aiq -t 06:59:00 -c 14 -g 1 -m 128G -a 0,3,7%4 -J evd-<sweep> \
#       evd-vfm/scripts/slurm/campagna.sbatch <sweep>
#
# The array index is the position in the WHOLE plan, not in the list of the missing cells, so
# a partial relaunch is a sparse array: renumbering would make an index point to another cell.
#
# --signal=B:USR1@300 is what saves the work: SLURM warns the process 300 s before the wall
# time, the training saves at the next step boundary and exits declaring the interruption.
# A relaunch of the same cell resumes from there; past 7 h chain the jobs with
# `hpc/invia.sh ... -d afterany:<previous job>`. A chain on finished cells costs nothing, since
# `campagna.py esegui --indice` asks the registry whether the cell is already complete.
#SBATCH --job-name=evd-campagna
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@300

set -euo pipefail

SWEEP="${1:-}"
[[ -n "$SWEEP" ]] || { echo "usage: $0 <sweep> (a file of evd-vfm/configs/sweep/)" >&2; exit 2; }

# The budget of this job comes from the Hydra config, not from the environment.
export EVD_BUDGET_DA_CONFIG=1

EVD_HPC_DIR="${EVD_HPC_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/hpc}"
# shellcheck source=../../../hpc/env.sh
source "$EVD_HPC_DIR/env.sh"
evd_prepare_dirs
evd_summary
cd "$EVD_ROOT/evd-vfm"

[[ -f "configs/sweep/$SWEEP.yaml" ]] || { echo "no such sweep: configs/sweep/$SWEEP.yaml" >&2; exit 2; }
: "${SLURM_ARRAY_TASK_ID:?a campaign is submitted as an array: hpc/invia.sh -a <indices>}"

# Extra arguments of `campagna.py esegui`: an array has no place for them. `--tutte` skips the
# registry check, the only way of evaluating again a complete cell.
read -r -a EVD_EXTRA <<< "${EVD_CAMPAGNA_EXTRA:-}"

# `exec`: SIGUSR1 reaches the foreground process, and bash in between has nothing to save.
exec "$EVD_PYTHON" scripts/campagna.py esegui "$SWEEP" --indice "$SLURM_ARRAY_TASK_ID" \
    ${EVD_EXTRA[@]+"${EVD_EXTRA[@]}"}
