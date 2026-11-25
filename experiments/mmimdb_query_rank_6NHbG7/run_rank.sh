#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo $? > job.exit' EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NCCL_P2P_DISABLE=1
py=/disk1/longw/conda_envs/mm/bin/python
root=/disk1/longw/work/MetaPrompt_code_1116
CUDA_VISIBLE_DEVICES=1,2 "$py" -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py 2>train.err
CUDA_VISIBLE_DEVICES=1 "$py" -u ../mmimdb_curriculum_20261119/evaluate_mmimdb.py --repo "$root" --checkpoint "$PWD/selected.ckpt" --method query_rank --output "$PWD/selected.csv" > selected.eval.log 2>&1 &
test_pid=$!
CUDA_VISIBLE_DEVICES=2 "$py" -u diagnose.py > probe.log 2>&1 &
probe_pid=$!
test_rc=0; probe_rc=0
wait "$test_pid" || test_rc=$?
wait "$probe_pid" || probe_rc=$?
if ((test_rc)); then tail -20 selected.eval.log; exit "$test_rc"; fi
if ((probe_rc)); then tail -20 probe.log; exit "$probe_rc"; fi
"$py" finalize.py
