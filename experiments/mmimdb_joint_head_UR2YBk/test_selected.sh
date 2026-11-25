#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo $? > test.exit' EXIT
root=/disk1/longw/work/MetaPrompt_code_1116
py=/disk1/longw/conda_envs/mm/bin/python
test -s selection.json
if test "$(readlink -f selected.ckpt)" = "$(readlink -f ../mmimdb_joint_full_9aNxqw/selected.ckpt)"; then
    ln -s ../mmimdb_joint_full_9aNxqw/selected.csv test.csv
else
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 "$py" -u ../mmimdb_curriculum_20261119/evaluate_mmimdb.py --repo "$root" --checkpoint "$PWD/selected.ckpt" --method joint_visual_head --output "$PWD/test.csv" > test.log 2>&1 || { tail -20 test.log; exit 1; }
fi
"$py" report.py
