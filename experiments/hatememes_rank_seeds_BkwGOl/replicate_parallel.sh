#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bundle=$PWD
root=${bundle%/experiments/*}
py=/disk1/longw/conda_envs/mm/bin/python
trap 'echo "$?" > "$bundle/bundle.exit"' EXIT
run_one() (
    set -e
    seed=$1
    cd "$bundle/seed$seed"
    trap 'echo "$?" > pipeline.exit' EXIT
    export FOOD_JOB="$PWD" METAPROMPT_SINGLE_LABEL_TASK=hatememes METAPROMPT_TRAIN_SEED="$seed"
    export CUDA_VISIBLE_DEVICES="$2" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH="$PWD:$root"
    "$py" - <<'PY'
import ctypes,os
from pathlib import Path
lib=ctypes.CDLL(None)
fd=lib.inotify_init1(os.O_CLOEXEC)
assert fd>=0 and lib.inotify_add_watch(fd,b".",0x8|0x80)>=0
p=Path("curriculum.exit")
while not p.exists() or not p.read_text().strip():
    os.read(fd,65536)
assert p.read_text().strip()=="0","Curriculum training failed"
os.close(fd)
PY
    for stage in prompt_meta candidate; do
        echo "$(date -Is) $stage" >> stages.log
        "$py" -u "$stage.py" > "$stage.log" 2>&1
    done
    echo "$(date -Is) rank_head" >> stages.log
    "$py" -u rank/rank_head.py --block text > rank/fit.log 2>&1
    echo "$(date -Is) test" >> stages.log
    "$py" -u rank/rank_test.py --replication > rank/test.log 2>&1
    echo "$(date -Is) complete" >> stages.log
)
run_one 1 5 &
p1=$!
run_one 2 6 &
p2=$!
a=0
b=0
wait "$p1" || a=$?
wait "$p2" || b=$?
if (( a!=0 || b!=0 )); then
    echo "seed 1 exit $a, seed 2 exit $b" >&2
    exit 1
fi
"$py" seed_summary.py
