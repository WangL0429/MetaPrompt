#!/usr/bin/env bash
set -euo pipefail
job=$(cd "$(dirname "$0")" && pwd)
cd "$job"
trap 'echo "$?" > queue.exit' EXIT
py=/disk1/longw/conda_envs/mm/bin/python
exp=$job/pipeline/experiments
base=$exp/mmimdb_joint_full_9aNxqw
sem=$exp/mmimdb_pair_ablation_dZfki9
export PYTHONPATH="$job/pipeline"
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NCCL_P2P_DISABLE=1
stage() {
    echo "$(date -Is) $1" >> stages.log
    local label=$1
    shift
    "$@" > "$job/$label.log" 2>&1
}
# The original trainer is detached. Block on its exit signal without polling logs.
"$py" - "$job" <<'WAIT'
import ctypes, os, sys
from pathlib import Path
job=Path(sys.argv[1])
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_CLOEXEC)
if fd < 0 or lib.inotify_add_watch(fd,os.fsencode(job),0x8 | 0x80) < 0:
    raise OSError(ctypes.get_errno(),"inotify")
exit_file=job/"curriculum.exit"
while not exit_file.exists() or not exit_file.read_text().strip():
    os.read(fd,65536)
os.close(fd)
assert (job/"curriculum.exit").read_text().strip()=="0"
WAIT
head=$exp/mmimdb_head_QeuxLr
stage features_best_train "$py" "$head/features.py" best train
stage features_best_dev "$py" "$head/features.py" best dev
stage features_last_dev "$py" "$head/features.py" last dev
stage fit_shared_head "$py" "$head/fit.py"
stage serialize_prompts "$py" "$exp/mmimdb_visual_prompt_NOSgpO/common.py" init
stage teacher_cache "$py" "$base/cache.py"
stage second_order env CUDA_VISIBLE_DEVICES=5,4 COURSE_ARM=original COURSE_SEED=1 "$py" -u -m torch.distributed.run --standalone --nproc_per_node=2 "$base/train.py"
stage projected_poster "$py" "$exp/mmimdb_poster_probe_S5KTyc/extract.py"
stage paired_features "$py" "$exp/mmimdb_joint_head_UR2YBk/extract_complete.py"
stage joint_head "$py" "$exp/mmimdb_joint_head_UR2YBk/train_head.py"
stage candidate_features "$py" "$sem/encode.py" train dev
stage single_candidate "$py" -u "$sem/fit_semantics.py" 1 --no-paired-supervision
stage fit_readout "$py" -u "$job/readout.py" fit
stage test_readout "$py" -u "$job/readout.py" test
echo "$(date -Is) complete" >> stages.log
