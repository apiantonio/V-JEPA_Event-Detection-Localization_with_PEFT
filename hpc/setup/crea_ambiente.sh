#!/usr/bin/env bash
# Build a private copy of the Python environment, pinned by requisiti-hpc.txt. Run it once on the
# login node. It is not needed to run the code: /shared/tesi_mllm/venv-hpc is already there.
#
#   bash hpc/setup/crea_ambiente.sh <target folder>
#
# Python 3.12 of the cluster has no ensurepip and PEP 668 blocks `pip install --user`, so the
# venv is created without pip and pip comes from get-pip.py. A venv is a path that works in a
# non-interactive job; conda would need `module`, which exists only in interactive shells.
# Then export EVD_VENV=<target folder> before sourcing hpc/env.sh.

set -euo pipefail

target="${1:?usage: crea_ambiente.sh <target folder>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ ! -e "$target" ] || { echo "ERROR: $target already exists" >&2; exit 2; }

python3.12 -m venv --without-pip "$target"
get_pip="$(mktemp)"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$get_pip"
"$target/bin/python" "$get_pip"
rm -f "$get_pip"

"$target/bin/python" -m pip install -r "$here/requisiti-hpc.txt"
"$target/bin/python" -c 'import torch; print("torch", torch.__version__)'
echo "environment ready: $target"
