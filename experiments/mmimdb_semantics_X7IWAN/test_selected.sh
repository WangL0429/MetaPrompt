#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
exec 9<training.lock
flock -s 9
test "$(cat training.exit)" = 0
trap 'echo $? > evaluation.exit' EXIT
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
py=/disk1/longw/conda_envs/mm/bin/python
CUDA_VISIBLE_DEVICES=0 "$py" evaluate_semantics.py 1 >k1/evaluate.log 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=5 "$py" evaluate_semantics.py 4 >k4/evaluate.log 2>&1 & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
if (( rc )); then
    tail -20 k1/evaluate.log k4/evaluate.log
    exit 1
fi
"$py" compare_semantics.py
