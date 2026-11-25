#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
py=/disk1/longw/conda_envs/mm/bin/python
"$py" -c 'import json; assert json.load(open("decision.json"))["replicate"]'
# Use two free GPUs per arm. If only one pair is available, reuse it in series.
readarray -t free < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, '$2+0<300 && $3+0<5 {print $1+0}')
if (( ${#free[@]} < 2 )); then echo NO_FREE_PAIR; exit 2; fi
export COURSE_SEED=1
base="$(basename "$PWD")"
run_arm() (
    arm="$1"
    out="../${base}_s1_${arm}"
    mkdir -p "$out"
    if ! test -e "$out/train.py"; then
        cp course.py train.py model_io.py rule.py report.py run.sh "$out/"
        ln -s ../mmimdb_joint_full_9aNxqw/train_cache.pt "$out/train_cache.pt"
        ln -s ../mmimdb_joint_full_9aNxqw/posters.npy "$out/posters.npy"
    fi
    cd "$out"
    if test -s job.exit && test "$(cat job.exit)" = 0; then echo "$arm complete"; exit; fi
    export COURSE_ARM="$arm" TRAIN_GPUS="$2"
    set -o pipefail
    bash run.sh | tee job.log
)
pair="${free[0]},${free[1]}"
if (( ${#free[@]} >= 4 )); then
    run_arm original "$pair" & a=$!
    run_arm early "${free[2]},${free[3]}" & b=$!
    bad=0
    wait "$a" || bad=1
    wait "$b" || bad=1
    test "$bad" = 0
else
    run_arm original "$pair"
    run_arm early "$pair"
fi
echo REPLICATION_DONE
