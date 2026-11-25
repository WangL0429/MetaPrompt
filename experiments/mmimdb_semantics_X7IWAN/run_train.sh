#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
exec 9>training.lock
flock -n 9 || exit 2
trap 'echo $? > training.exit' EXIT
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
py=/disk1/longw/conda_envs/mm/bin/python
mkdir -p k1 k4
CUDA_VISIBLE_DEVICES=0 "$py" fit_semantics.py 1 >k1/train.log 2>&1 & a=$!
CUDA_VISIBLE_DEVICES=5 "$py" fit_semantics.py 4 >k4/train.log 2>&1 & b=$!
rc=0
wait "$a" || rc=1
wait "$b" || rc=1
exit "$rc"
