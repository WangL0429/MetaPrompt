#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo $? > test.exit' EXIT
py=/disk1/longw/conda_envs/mm/bin/python
root=/disk1/longw/work/MetaPrompt_code_1116
test -s selection.json
if test -L selected.ckpt; then
    source=$("$py" -c 'import json; print(json.load(open("run_config.json"))["source"])')
    ln -s "../$source/selected.csv" test.csv
else
    "$py" -u ../mmimdb_curriculum_20261119/evaluate_mmimdb.py --repo "$root" --checkpoint "$PWD/selected.ckpt" --method constrained_visual_head --output "$PWD/test.csv" > test.log 2>&1 || { tail -20 test.log; exit 1; }
fi
