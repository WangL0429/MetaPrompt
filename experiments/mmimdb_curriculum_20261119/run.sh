#!/usr/bin/env bash
set -u
cd /disk1/longw/work/MetaPrompt_code_1116 || exit 1
job=experiments/mmimdb_curriculum_20261119
python=/disk1/longw/conda_envs/mm/bin/python
export OMP_NUM_THREADS=1

run_arm() {
    local arm=$1 gpu=$2 status
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u run.py with task_finetune_mmimdb \
        data_root=/disk1/longw/Datasets/mmimdb mmimdb_meta=False \
        train_missing_schedule="$arm" max_epoch=20 per_gpu_batchsize=32 \
        num_workers=8 val_check_interval=1.0 log_dir="$job/checkpoints" \
        exp_name="$arm" > "$job/$arm.train.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$job/$arm.train.exit"
    return "$status"
}

run_arm curriculum 1 &
course_pid=$!
run_arm random 3 &
random_pid=$!
wait "$course_pid"
course_status=$?
wait "$random_pid"
random_status=$?
if ((course_status != 0 || random_status != 0)); then
    printf 'curriculum=%d random=%d\n' "$course_status" "$random_status"
    printf '1\n' > "$job/job.exit"
    exit 1
fi

bash "$job/evaluate.sh" > "$job/evaluation.log" 2>&1
status=$?
printf '%s\n' "$status" > "$job/job.exit"
exit "$status"
