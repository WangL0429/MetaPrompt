#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
root=/disk1/longw/work/MetaPrompt_code_1116
py=/disk1/longw/conda_envs/mm/bin/python
export OMP_NUM_THREADS=2
run_arm() {
    local arm=$1 gpu=$2
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$py" -u train_meta.py "$arm" > "$arm.train.log" 2>&1 || return $?
    "$py" -u ../mmimdb_curriculum_20261119/evaluate_mmimdb.py --repo "$root" --checkpoint "$PWD/$arm.ckpt" --method "$arm" --output "$PWD/$arm.csv" > "$arm.eval.log" 2>&1
}
run_arm equal 0 & p1=$!
run_arm fomaml 1 & p2=$!
run_arm learned 3 & p3=$!
failed=0
for pid in "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
if ((failed==0)); then "$py" report.py || failed=1; fi
printf '%s\n' "$failed" > job.exit
exit "$failed"
