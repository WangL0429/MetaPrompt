#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=5 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
py=/disk1/longw/conda_envs/mm/bin/python
"$py" - <<'PY'
# Block on kernel notifications while the parallel cache writers finish.
import ctypes, os
from pathlib import Path
lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(0)
assert fd>=0 and lib.inotify_add_watch(fd,b".",8)>=0
status=Path("cache.exit")
while not status.exists() or not status.read_text().strip():
    os.read(fd,8192)
assert status.read_text().strip()=="0"
PY
"$py" train_last_prompt.py >joint/train.log 2>&1
"$py" evaluate_last_prompt.py >joint/test.log 2>&1
"$py" compare_last_prompt.py
