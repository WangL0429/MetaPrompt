#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec 9>queue.lock
flock -n 9 || exit 73
trap 'echo $? > queue.exit' EXIT
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-4,5}"
export CAP_RANK_0="${CAP_RANK_0:-0.49}" CAP_RANK_1="${CAP_RANK_1:-0.31}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NCCL_P2P_DISABLE=1
py=/disk1/longw/conda_envs/mm/bin/python
root=/disk1/longw/work/MetaPrompt_code_1116
"$py" protocol.py --repo "$root" > mask_audit.tmp
mv mask_audit.tmp mask_audit.json
for seed in 0 1; do
    for arm in course_shared high_shared high_text high_image; do
        out="s${seed}_${arm}"
        mkdir -p "$out"
        if ! test -s "$out/selection.json"; then
            echo "Training $out" >> phases.log
            "$py" -u -m torch.distributed.run --standalone --nproc_per_node=2 train_matched.py "$arm" "$seed" >> "$out/train.log" 2>&1 || { tail -55 "$out/train.log"; exit 1; }
        fi
    done
done
IFS=, read -r gpu_a gpu_b <<< "$CUDA_VISIBLE_DEVICES"
for seed in 0 1; do
    for arm in course_shared high_shared high_text high_image; do
        out="s${seed}_${arm}"
        echo "Evaluating $out" >> phases.log
        if test "$(readlink -f "$out/last.ckpt")" = "$(readlink -f "$out/selected.ckpt")"; then
            CUDA_VISIBLE_DEVICES="$gpu_a" "$py" -u evaluate.py "$out/last" >> "$out/eval.log" 2>&1
            ln -sfn last.npz "$out/selected.npz"
        else
            CUDA_VISIBLE_DEVICES="$gpu_a" "$py" -u evaluate.py "$out/last" >> "$out/eval-last.log" 2>&1 & a=$!
            CUDA_VISIBLE_DEVICES="$gpu_b" "$py" -u evaluate.py "$out/selected" >> "$out/eval-selected.log" 2>&1 & b=$!
            rc=0; wait "$a" || rc=1; wait "$b" || rc=1
            if (( rc )); then tail -25 "$out"/eval-*.log; exit 1; fi
        fi
    done
done
"$py" -u report.py
