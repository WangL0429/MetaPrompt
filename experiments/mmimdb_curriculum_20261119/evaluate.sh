#!/usr/bin/env bash
set -u
root=/disk1/longw/work/MetaPrompt_code_1116
job="$root/experiments/mmimdb_curriculum_20261119"
dpl=/disk1/longw/work/mm_sota/DPL/DPL_DCP
mask_root=/disk1/longw/work/mm_sota/missing_grid_20261110/masks
dpl_python=/home/wanglong/anaconda3/envs/MM_DCP/bin/python
python=/disk1/longw/conda_envs/mm/bin/python
export OMP_NUM_THREADS=1

evaluate_current() {
    local arm=$1 gpu=$2
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u "$job/evaluate_mmimdb.py" \
        --repo "$root" --method "$arm" --output "$job/$arm.csv" \
        --checkpoint "$job/checkpoints/${arm}_seed0/version_0/checkpoints/last.ckpt" \
        > "$job/$arm.eval.log" 2>&1
}
evaluate_current curriculum 1 &
course_pid=$!
evaluate_current random 3 &
random_pid=$!
failed=0
wait "$course_pid" || failed=1
wait "$random_pid" || failed=1
if ((failed)); then
    exit 1
fi

(
    cd "$dpl" || exit 1
    CUDA_VISIBLE_DEVICES=1 "$dpl_python" -u grid_eval_20261110/dpl_grid.py         --manifest "$job/dpl_specialists.json" --output "$job/dpl_specialists"         --mask-root "$mask_root" --workers 8
) > "$job/dpl_specialists.eval.log" 2>&1 || exit $?
"$python" -u "$job/summarize_mmimdb.py"
