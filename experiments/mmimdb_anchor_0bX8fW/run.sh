#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
py=/disk1/longw/conda_envs/mm/bin/python
root=/disk1/longw/work/MetaPrompt_code_1116
export OMP_NUM_THREADS=2
run_arm() (
    set -e -o pipefail
    arm=$1
    export CUDA_VISIBLE_DEVICES="$2"
    "$py" -u train_anchor.py "$arm" 2> "$arm.train.err" | tee "$arm.train.log"
    if test -L "$arm.ckpt"; then
        ln -s ../mmimdb_visual_prompt_NOSgpO/initial.csv "$arm.csv"
    else
        "$py" -u ../mmimdb_curriculum_20261119/evaluate_mmimdb.py --repo "$root" --checkpoint "$PWD/$arm.ckpt" --method "$arm" --output "$PWD/$arm.csv" > "$arm.eval.log" 2>&1
    fi
)
run_arm control 0 & left=$!
run_arm anchor 1 & right=$!
rc=0
wait "$left" || rc=1
wait "$right" || rc=1
if (( rc == 0 )); then "$py" report_anchor.py || rc=1; fi
if (( rc != 0 )); then tail -20 ./*.err ./*.eval.log 2>/dev/null; fi
echo "$rc" > job.exit
exit "$rc"
