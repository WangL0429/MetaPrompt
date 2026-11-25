#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
py=/disk1/longw/conda_envs/mm/bin/python
CUDA_VISIBLE_DEVICES=0 "$py" run_grid_objective.py control >control/run.log 2>&1 & control=$!
CUDA_VISIBLE_DEVICES=5 "$py" run_grid_objective.py grid >grid/run.log 2>&1 & grid=$!
rc=0
wait "$control" || rc=1
wait "$grid" || rc=1
if (( rc )); then
    tail -20 control/run.log grid/run.log
    exit 1
fi
CUDA_VISIBLE_DEVICES=5 "$py" compare_grid_objective.py
