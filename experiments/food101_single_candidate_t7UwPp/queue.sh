#!/usr/bin/env bash
set -eu -o pipefail
cd "$(dirname "$0")"
export FOOD_JOB="$PWD"
export PYTHONPATH="$PWD:${PWD%/experiments/*}"
export CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
py=/disk1/longw/conda_envs/mm/bin/python
trap 'echo "$?" > queue.exit' EXIT
"$py" - <<'PY'
import ctypes, os
from pathlib import Path
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_CLOEXEC)
assert fd>=0 and lib.inotify_add_watch(fd,b".",0x8 | 0x80)>=0
p=Path("curriculum.exit")
while not p.exists() or not p.read_text().strip():
    os.read(fd,65536)
assert p.read_text().strip()=="0"
os.close(fd)
PY
for stage in prompt_meta candidate evaluate; do
    echo "$(date -Is) $stage" >> stages.log
    "$py" -u "$stage.py" > "$stage.log" 2>&1
done
echo "$(date -Is) complete" >> stages.log
