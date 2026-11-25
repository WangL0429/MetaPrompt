#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'echo $? > job.exit' EXIT
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS:-0,1}"
IFS=, read -r first_gpu second_gpu <<< "$CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=2
export NCCL_P2P_DISABLE=1
py=/disk1/longw/conda_envs/mm/bin/python
root=/disk1/longw/work/MetaPrompt_code_1116
"$py" -u -m torch.distributed.run --standalone --nproc_per_node=2 train.py 2> train.err
do_eval() {
    CUDA_VISIBLE_DEVICES="$2" "$py" -u ../mmimdb_curriculum_20261119/evaluate_mmimdb.py --repo "$root" --checkpoint "$PWD/$1.ckpt" --method "$1" --output "$PWD/$1.csv" > "$1.eval.log" 2>&1
}
do_eval last "$first_gpu" & lastpid=$!
if test -L selected.ckpt; then
    if test "$(readlink -f selected.ckpt)" = "$(readlink -f last.ckpt)"; then
        target=last.csv
    else
        target=../mmimdb_visual_prompt_NOSgpO/initial.csv
    fi
    ln -sf "$target" selected.csv
else
    do_eval selected "$second_gpu" & chosenpid=$!
    wait "$chosenpid"
fi
wait "$lastpid"
"$py" report.py
