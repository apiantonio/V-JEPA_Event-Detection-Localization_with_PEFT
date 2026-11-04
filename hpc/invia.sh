#!/usr/bin/env bash
# hpc/invia.sh: submit a job, checking first what SLURM reports badly or late.
#
#   hpc/invia.sh [options] <job.sbatch> [arguments of the job]
#
# Options
#   -p <partition>  aiq (H100), gpuq (A100) or fatq (CPU only). Default: $EVD_PARTITION.
#   -t <time>       wall time: "30" (minutes), "7:00:00", ... aiq and gpuq stop at 7 h, fatq at 4 h.
#   -c <n>          CPUs per task.
#   -g <n>          GPUs per task; 0 on fatq.
#   -m <mem>        memory per node. Default: 96G on aiq and gpuq, the whole node on fatq.
#                   Without --mem SLURM reserves the WHOLE node, and the job waits for an empty one.
#   -A <account>    SLURM account. Default: did_tesi_mllm_433 (the job is rejected without it).
#   -J <name>       job name, also the prefix of the log files.
#   -a <spec>       job array: "0-7", "0,3,7", "0-7%4". Every task counts against the limit of
#                   10 jobs submitted per user, whatever the "%n".
#   -d <dependency> e.g. "afterany:812345". Past 7 h a training continues with a chain of
#                   afterany jobs; take the id from the output of the previous submission.
#   -R <n>          requeue on node failure or preemption. It does not extend the wall time.
#   -n              dry run: print the sbatch command and submit nothing.
#   -h              this text.

set -euo pipefail

EVD_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./env.sh
source "$EVD_HPC_DIR/env.sh"

MAX_ARRAY_INDEX=1000
MAX_SUBMIT=${EVD_SLURM_MAX_SUBMIT:-10}

partition_max_seconds() {
    case "$1" in
        aiq|gpuq) echo 25200 ;;
        fatq) echo 14400 ;;
        *) echo "" ;;
    esac
}

partition_max_requeue() {
    case "$1" in
        aiq|gpuq) echo 68 ;;
        fatq) echo 120 ;;
        *) echo "" ;;
    esac
}

# "7:00:00", "30" (minutes), "10:30" (mm:ss), "1-02:00:00" -> seconds; empty if not understood.
to_seconds() {
    local t="$1" days=0 h=0 m=0 s=0 field
    case "$t" in
        *-*) days="${t%%-*}"; t="${t#*-}" ;;
    esac
    case "$t" in
        *:*:*) IFS=: read -r h m s <<<"$t" ;;
        *:*)   IFS=: read -r m s <<<"$t" ;;
        *)     m="$t" ;;
    esac
    for field in "$days" "$h" "$m" "$s"; do
        case "$field" in
            "" | *[!0-9]*) echo ""; return ;;
        esac
    done
    echo $(( 10#$days * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$s ))
}

help_text() { awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

partition="$EVD_PARTITION"
wall_time="$EVD_TIME"
cpus="$EVD_CPUS_PER_TASK"
gpus="$EVD_GPUS_PER_TASK"
mem="$EVD_MEM_PER_NODE"
account="$EVD_SLURM_ACCOUNT"
name=""
array=""
dependency=""
requeue=""
dry_run=0

while getopts ":p:t:c:g:m:A:J:a:d:R:nh" opt; do
    case "$opt" in
        p) partition="$OPTARG" ;;
        t) wall_time="$OPTARG" ;;
        c) cpus="$OPTARG" ;;
        g) gpus="$OPTARG" ;;
        m) mem="$OPTARG" ;;
        A) account="$OPTARG" ;;
        J) name="$OPTARG" ;;
        a) array="$OPTARG" ;;
        d) dependency="$OPTARG" ;;
        R) requeue="$OPTARG" ;;
        n) dry_run=1 ;;
        h) help_text; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; exit 2 ;;
        :)  echo "option -$OPTARG needs a value" >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -lt 1 ]; then
    echo "usage: hpc/invia.sh [options] <job.sbatch> [arguments of the job]; -h lists the options" >&2
    exit 2
fi

job_script="$1"; shift
[ -f "$job_script" ] || { echo "ERROR: job script not found: '$job_script'" >&2; exit 2; }
job_script="$(cd "$(dirname "$job_script")" && pwd)/$(basename "$job_script")"
[ -n "$name" ] || name="evd-$(basename "$job_script" .sbatch)"

errors=0
fail() { echo "ERROR: $*" >&2; errors=$((errors + 1)); }

case "$partition" in
    aiq|gpuq|fatq) ;;
    *) fail "partition '$partition' has no QOS for this account: use aiq, gpuq or fatq." ;;
esac

limit="$(partition_max_seconds "$partition")"
requested="$(to_seconds "$wall_time")"
if [ -n "$limit" ] && [ -n "$requested" ] && [ "$requested" -gt "$limit" ]; then
    fail "--time=$wall_time is over the $((limit / 3600)) h limit of $partition: the job would wait forever."
fi

if [ -z "$mem" ]; then
    if [ "$partition" = "fatq" ]; then mem=0; else mem=96G; fi
fi
if [ "$mem" != "0" ] && ! [[ "$mem" =~ ^[0-9]+[KMGTkmgt]?$ ]]; then
    fail "-m wants a size such as 96G, 98304M or 0, not '$mem'."
fi

if [ "${gpus:-0}" -gt 0 ] && [ "$partition" = "fatq" ]; then
    fail "$gpus GPUs requested on fatq, which has none: pass -g 0."
fi
if [ "${gpus:-0}" -eq 0 ] && [ "$partition" != "fatq" ]; then
    echo "warning: no GPU on $partition: for CPU work use -p fatq and leave the cards free." >&2
fi
[ -n "$account" ] || fail "no SLURM account: the job would be rejected."

if [ -n "$array" ]; then
    highest="$(printf '%s\n' "${array%%\%*}" | tr ',-' '\n\n' | grep -E '^[0-9]+$' | sort -n | tail -1)"
    if [ -n "$highest" ] && [ "$highest" -gt "$MAX_ARRAY_INDEX" ]; then
        fail "array index $highest is over the limit of the cluster (0..$MAX_ARRAY_INDEX)."
    fi
    n_tasks="$(
        printf '%s\n' "${array%%\%*}" | tr ',' '\n' | awk -F- '
            NF == 2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ { n += $2 - $1 + 1; next }
            $0 ~ /^[0-9]+$/ { n += 1 }
            END { print n + 0 }'
    )"
    queued="$(squeue -u "$USER" -h -r 2>/dev/null | wc -l | tr -d ' ')"
    free=$((MAX_SUBMIT - ${queued:-0}))
    if [ "$n_tasks" -gt "$free" ]; then
        fail "array of $n_tasks tasks, but only $free can be submitted (limit $MAX_SUBMIT, ${queued:-0} queued); SLURM would answer AssocMaxSubmitJobLimit."
    fi
fi

if [ -n "$requeue" ]; then
    max_requeue="$(partition_max_requeue "$partition")"
    if ! [[ "$requeue" =~ ^[0-9]+$ ]] || [ "$requeue" -lt 1 ]; then
        fail "-R wants an integer >= 1, not '$requeue'."
    elif [ -n "$max_requeue" ] && [ "$requeue" -gt "$max_requeue" ]; then
        fail "-R $requeue is over the limit of $partition ($max_requeue)."
    fi
fi

# A payload that does not exist would get its resources and die in two seconds.
if [ $# -ge 1 ]; then
    case "$1" in
        -*) ;;
        *.py | */*)
            if [ ! -f "$1" ] && [ ! -f "$EVD_ROOT/$1" ] && [ ! -f "$EVD_ROOT/evd-vfm/$1" ]; then
                fail "the job would run '$1', which does not exist (paths are relative to evd-vfm/)."
            fi
            ;;
    esac
fi

[ "$errors" -eq 0 ] || exit 1

evd_prepare_dirs

export_spec="ALL"
[ -n "$requeue" ] && export_spec="ALL,NREQUEUE=$requeue"

args=(
    --partition="$partition"
    --job-name="$name"
    --time="$wall_time"
    --ntasks=1
    --cpus-per-task="$cpus"
    --chdir="$EVD_ROOT"
    --open-mode=append
    --export="$export_spec"
    # Always explicit: omitting an option does not cancel an #SBATCH directive, and only
    # --gpus-per-task=0 overrides one.
    --gpus-per-task="${gpus:-0}"
    --mem="$mem"
    --account="$account"
)
[ -n "$dependency" ] && args+=(--dependency="$dependency")
[ -n "$requeue" ] && args+=(--requeue)
if [ -n "$array" ]; then
    args+=(--array="$array" --output="$EVD_LOG_DIR/%x-%A_%a.out" --error="$EVD_LOG_DIR/%x-%A_%a.err")
else
    args+=(--output="$EVD_LOG_DIR/%x-%j.out" --error="$EVD_LOG_DIR/%x-%j.err")
fi

echo "sbatch ${args[*]} $job_script $*"
if [ "$dry_run" -eq 1 ]; then
    echo "(dry run: not submitted)"
    exit 0
fi

outcome="$(sbatch "${args[@]}" "$job_script" "$@")"
echo "$outcome"
job_id="$(echo "$outcome" | grep -oE '[0-9]+$' || true)"
if [ -n "$job_id" ]; then
    cat <<EOF

  state:   squeue -j $job_id
  log:     tail -f $EVD_LOG_DIR/$name-$job_id.out
  cancel:  scancel $job_id
  after:   sacct -j $job_id --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES%40
EOF
fi
