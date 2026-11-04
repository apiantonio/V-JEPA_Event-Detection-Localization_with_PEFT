#!/bin/bash
# Run a Python script of evd-vfm/ with the resources given to hpc/invia.sh, for example:
#
#   hpc/invia.sh -p fatq -g 0 -c 4 -m 32G -t 00:20:00 hpc/jobs/python.sbatch scripts/smoke.py --strict
#
# Paths are relative to evd-vfm/. For training use evd-vfm/scripts/slurm/campagna.sbatch, which
# also asks for the signal that saves a checkpoint before the wall time.
#SBATCH --job-name=evd-python
#SBATCH --ntasks=1

set -euo pipefail

EVD_HPC_DIR="${EVD_HPC_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/hpc}"
# shellcheck source=../env.sh
source "$EVD_HPC_DIR/env.sh"
evd_prepare_dirs
evd_summary

[ $# -gt 0 ] || { echo "ERROR: no script to run" >&2; exit 2; }
cd "$EVD_ROOT/evd-vfm"
echo "=== $EVD_PYTHON $*"
exec "$EVD_PYTHON" "$@"
