#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 2
trap 'echo $? > queue.exit' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
py=/disk1/longw/conda_envs/mm/bin/python
test -s training_pairs.pt
test -s pair_audit.json
run_arm() {
    local arm="$1" gpu="$2"
    mkdir -p "$arm"
    if ! CUDA_VISIBLE_DEVICES="$gpu" "$py" train_ab.py "$arm" > "$arm/train.log" 2>&1; then
        if grep -qi 'out of memory' "$arm/train.log"; then
            CUDA_VISIBLE_DEVICES="$gpu" PAIR_MICRO=8 "$py" train_ab.py "$arm" >> "$arm/train.log" 2>&1
        else
            tail -25 "$arm/train.log"
            return 1
        fi
    fi
}
run_arm A 1 &
a_pid=$!
run_arm B 5 &
b_pid=$!
a_rc=0; b_rc=0
wait "$a_pid" || a_rc=$?
wait "$b_pid" || b_rc=$?
test "$a_rc" = 0 && test "$b_rc" = 0 || exit 1
CUDA_VISIBLE_DEVICES=1 "$py" assess.py A > A/assessment.log 2>&1 &
a_pid=$!
CUDA_VISIBLE_DEVICES=5 "$py" assess.py B > B/assessment.log 2>&1 &
b_pid=$!
a_rc=0; b_rc=0
wait "$a_pid" || a_rc=$?
wait "$b_pid" || b_rc=$?
test "$a_rc" = 0 && test "$b_rc" = 0 || exit 1
"$py" summarize.py
