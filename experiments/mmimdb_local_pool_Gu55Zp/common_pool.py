import csv
import json
import pickle
import sys
from pathlib import Path

import torch

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
BASE=EXP/'mmimdb_joint_full_9aNxqw'
SOURCE=EXP/'mmimdb_joint_head_UR2YBk'/'selected.ckpt'
sys.path.insert(0,str(BASE))
from model_io import load,plots,clip
from common import val_grid
from evaluate_mmimdb import MODES,RATES,SEEDS,load_test_masks


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False


def metadata():
    with (EXP/'mmimdb_matched_probe_obZdDo'/'metadata.pkl').open('rb') as f:
        return pickle.load(f)


def write(name,rows):
    path=JOB/name
    temp=path.with_suffix('.tmp')
    with temp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)
