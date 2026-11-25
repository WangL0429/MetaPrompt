"""Frozen CLIP features and spatial / textual summaries shared by both conditional models."""
import csv
import json
import pickle
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
BASE=EXP/"mmimdb_joint_full_9aNxqw"
SOURCE=EXP/"mmimdb_joint_head_UR2YBk/selected.ckpt"
sys.path[:0]=[str(BASE),str(EXP/"mmimdb_local_pool_Gu55Zp")]
from model_io import load,plots,choose_weights
from common import val_grid
from pool import Capture,projectors
from clip.datamodules.missing_schedule import make_schedule
from evaluate_mmimdb import MODES,RATES,SEEDS,load_test_masks


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False


def metadata():
    with (EXP/"mmimdb_matched_probe_obZdDo/metadata.pkl").open("rb") as f:return pickle.load(f)


def store(path,value):
    temp=path.with_suffix(".part")
    torch.save(value,temp)
    temp.replace(path)


def csv_out(path,rows):
    with path.with_suffix(".part").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(".part").replace(path)


def json_out(path,value):
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def summarize_tokens(net,cap,state,idx=slice(None)):
    x,valid=cap.local(state,idx)
    if state==1:
        side=int(x.shape[1]**.5)
        assert side*side==x.shape[1]
        regions=F.adaptive_avg_pool2d(x.transpose(1,2).reshape(len(x),x.size(2),side,side).float(),(3,3))
        tokens=regions.flatten(2).transpose(1,2)
        mask=torch.ones(tokens.shape[:2],dtype=torch.bool,device=x.device)
    else:
        length=valid.sum(1,keepdim=True).clamp_min(1)
        bins=(torch.arange(x.shape[1],device=x.device)[None,:]*9//length).clamp_max(8)
        membership=F.one_hot(bins,9).transpose(1,2).float()*valid[:,None,:]
        count=membership.sum(-1)
        tokens=(membership/count.clamp_min(1)[:,:,None]) @ x.float()
        mask=count>0
    tokens=tokens.half() @ projectors(net)[state-1].half()
    return tokens.half(),mask
