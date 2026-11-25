"""Shared cached state layout and deployment-equivalent inference for the residual comparison."""
import csv
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

JOB = Path(__file__).resolve().parent
EXP = JOB.parent
CACHE = EXP / "mmimdb_proj_swap_80Obmt"
SOURCE = EXP / "mmimdb_joint_head_UR2YBk/selected.ckpt"
KEYS = ("model.image_encoder.proj", "model.text_encoder.text_projection")
HEAD = "mmimdb_classifier"
sys.path.insert(0, str(EXP/"mmimdb_head_QeuxLr"))
from fit import SCENARIOS, val_grid
from evaluate_mmimdb import MODES, RATES, SEEDS, load_test_masks


def write_csv(name, rows):
    path = JOB / name
    with path.with_suffix(".tmp").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=rows[0])
        out.writeheader()
        out.writerows(rows)
    path.with_suffix(".tmp").replace(path)


def setup():
    torch.set_num_threads(3)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False


def read(split):
    return {k:v.cuda() for k,v in torch.load(CACHE/f"lowrank_{split}.pt",map_location="cpu").items()}


def parameters(state):
    p = tuple(state[k].cuda() for k in KEYS)
    head = tuple(state[HEAD+"."+k].cuda().half() for k in ("weight","bias"))
    return p,head


def views(data):
    rows=data["rows"]
    return ((data["v0"][rows],data["t0"]), (data["v1"][rows],None), (None,data["t2"]))


def project(view, matrices):
    output = [h.half() @ p.half() if h is not None else None for h,p in zip(view,matrices)]
    if output[0] is None:
        output[0] = torch.zeros_like(output[1])
    if output[1] is None:
        output[1] = torch.zeros_like(output[0])
    return torch.cat(output,1)


def decode(view, matrices, head):
    return F.linear(project(view,matrices),*head).float()


def logits(data, matrices, head):
    blocks=[]
    for view in views(data):
        chunks=[]
        for i in range(0,len(data["y"]),64):
            batch=tuple(None if h is None else h[i:i+64] for h in view)
            chunks.append(decode(batch,matrices,head))
        blocks.append(torch.cat(chunks))
    return torch.stack(blocks)


def grid(z, data, keep):
    return val_grid(z[:,keep].cpu(),data["y"][keep].cpu())


def scenario_columns(values):
    return {f"{mode}_{round(rate*100)}":value for (mode,rate),value in zip(SCENARIOS,values)}
