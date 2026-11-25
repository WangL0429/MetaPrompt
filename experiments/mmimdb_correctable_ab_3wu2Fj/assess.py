"""Evaluate models selected on calibration films using pristine validation and archived test masks."""
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
BASE=EXP/"mmimdb_joint_full_9aNxqw"
SOURCE=EXP/"mmimdb_joint_head_UR2YBk/selected.ckpt"
CACHED=EXP/"mmimdb_joint_residual_BO9efZ"
sys.path.insert(0,str(BASE))
from model_io import load,plots


def baseline(split):
    if split=="dev":
        d=torch.load(CACHED/"pre_validation.pt",map_location="cpu")
        return d["scores"]["base"],d["rows"],d["y"]
    d=torch.load(CACHED/"test_predictions.pt",map_location="cpu")
    return d["outputs"]["base"],d["rows"],d["y"]


@torch.inference_mode()
def run(arm):
    torch.set_num_threads(2)
    root=JOB/arm
    with (EXP/"mmimdb_matched_probe_obZdDo/metadata.pkl").open("rb") as f:meta=pickle.load(f)
    outputs={"dev":{},"test":{}}
    cache={}
    for name in ("base","selected","trained"):
        path=(SOURCE if name=="base" else root/(name+".ckpt")).resolve(strict=True)
        if path==SOURCE.resolve():
            for split in outputs:outputs[split][name]=baseline(split)[0]
            continue
        if path in cache:
            for split in outputs:outputs[split][name]=outputs[split][cache[path]]
            continue
        net,ck=load(path)
        cfg=ck["hyper_parameters"]["config"]
        original=torch.load(SOURCE,map_location="cpu")["state_dict"]
        state=ck["state_dict"]
        assert set(original)==set(state)
        changed=[k for k in state if not torch.equal(original[k],state[k])]
        assert changed and all(k.startswith("model.prompt_learner.") for k in changed)
        del original,ck,state
        net.cuda().eval().requires_grad_(False)
        for split in outputs:
            ds=plots(cfg,split)
            m=meta[split]
            ids=[str(x).zfill(7) for x in ds.table["image_id"].to_pylist()]
            rows=np.asarray([ds.index_mapper[i][0] for i in range(len(ds))])
            assert ids==m["ids"] and np.array_equal(rows,m["rows"])
            ix=np.flatnonzero(~m["cal"][rows]) if split=="dev" else np.arange(len(ds))
            _,arch_rows,arch_y=baseline(split)
            assert np.array_equal(rows[ix],arch_rows.numpy()) and np.array_equal(m["labels"][rows[ix]],arch_y.numpy())
            loader=DataLoader(Subset(ds,ix.tolist()),batch_size=64,num_workers=4,pin_memory=True)
            chunks=[[],[],[]]
            with torch.autocast("cuda",dtype=torch.float16):
                for images,texts in loader:
                    images=images.cuda()
                    for mode in (0,1,2):
                        batch=dict(image=[torch.zeros_like(images) if mode==2 else images],
                                   text=([""]*len(images) if mode==1 else texts,),missing_type=[mode]*len(images))
                        z=net.mmimdb_classifier(net.infer(batch)["cls_feats"]).float()
                        assert z.isfinite().all()
                        chunks[mode].append(z.cpu())
            outputs[split][name]=torch.stack([torch.cat(x) for x in chunks])
        cache[path]=name
        del net
    tmp=root/"predictions.tmp"
    torch.save(outputs,tmp)
    tmp.replace(root/"predictions.pt")


if __name__=="__main__":run(sys.argv[1])
