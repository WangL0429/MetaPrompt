"""Frozen-feature attribution and projection conditioning for the MMIMDb single-candidate checkpoint."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SWAP = EXP/"mmimdb_proj_swap_80Obmt"
MATCH = EXP/"mmimdb_matched_probe_obZdDo"
MAIN = EXP/"mmimdb_pair_ablation_dZfki9"
sys.path.insert(0, str(MATCH))
from compare import ap, counts, fscore, logits, thresholds
from probe_fit import fit

SEEDS = (42,2026,3407)
RIDGES = (.0003,.003,.03,.3,3.,30.,300.)


def write(name,rows):
    with (HERE/name).open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def read(name):
    with name.open() as f:
        return list(csv.DictReader(f))


def metadata():
    with (MATCH/"metadata.pkl").open("rb") as f:
        return pickle.load(f)


def spectrum_and_probes(meta):
    wt = {s:np.bincount(v["rows"]) for s,v in meta.items() if s in ("train","dev","test")}
    weights = wt["train"]/wt["train"].sum()
    cal = meta["dev"]["cal"]
    y = {s:meta[s]["labels"] for s in wt}
    condition,features,probes,candidates = [],[],[],[]
    for arm in ("shared","dpl"):
        matrix = np.load(SWAP/f"{arm}_matrix.npy").astype(float)
        u,sv,_ = np.linalg.svd(matrix,full_matrices=True)
        rank = np.linalg.matrix_rank(matrix)
        assert rank==512 and matrix.shape==(768,512)
        with np.load(SWAP/f"{arm}_features.npz") as cache:
            raw = {s:cache[f"{s}_raw"].astype(float) for s in wt}
            post = {s:cache[f"{s}_native"].astype(float) for s in wt}
        centered=raw["train"]-weights@raw["train"]
        null = {s:raw[s]@u[:,rank:] for s in wt}
        null_fraction=(weights@np.square(centered@u[:,rank:]).sum(1))/(weights@np.square(centered).sum(1))
        condition.append(dict(model=arm,rank=rank,null_dim=matrix.shape[0]-rank,sigma_max=sv[0],
                              sigma_median=np.median(sv),sigma_min=sv[-1],condition_nonzero=sv[0]/sv[-1],
                              null_fraction_centered_train_variance=null_fraction))
        for variant,ds in (("native_white",post),("raw_white",raw),("null_white",null)):
            x=ds["train"]
            mean=weights@x
            centered=x-mean
            eig,rotation=np.linalg.eigh((centered.T*weights)@centered)
            floor=eig[-1]*1e-10
            whitening=rotation/np.sqrt(np.maximum(eig,floor))
            transformed={s:((array-mean)@whitening).astype(np.float32) for s,array in ds.items()}
            features.append(dict(model=arm,variant=variant,dimension=x.shape[1],
                                 eig_max=eig[-1],eig_min=eig[0],eig_floor=floor,
                                 floored=int((eig<floor).sum()),effective_dim=eig.sum()**2/(eig@eig)))
            if arm=="dpl" and variant!="native_white":
                continue
            data=dict(x=torch.from_numpy(transformed["train"]),labels=torch.from_numpy(y["train"]),
                      weight=torch.from_numpy(wt["train"]))
            trials=[]
            for ridge in RIDGES:
                w,b,grad=fit(data,torch.zeros(23,x.shape[1]),torch.zeros(23),ridge)
                z=logits(transformed["dev"][cal],w,b)
                score=ap(y["dev"][cal],z,wt["dev"][cal])
                trials.append((score,ridge,w,b,z))
                candidates.append(dict(model=arm,variant=variant,ridge=ridge,cal_AP=score,gradient=grad))
            score,ridge,w,b,z=max(trials,key=lambda r:(r[0],r[1]))
            cut=thresholds(y["dev"][cal],z,wt["dev"][cal])
            for s,ix in (("dev",~cal),("test",slice(None))):
                scores=logits(transformed[s][ix],w,b)
                pred=scores.astype(float)>cut
                probes.append(dict(model=arm,variant=variant,split=s,ridge=ridge,cal_AP=score,
                        pure_AP=ap(y[s][ix],scores,wt[s][ix]),
                        pure_F1=fscore(counts(y[s][ix],pred,wt[s][ix]).sum(0))))
                if arm=="shared" and s=="test" and variant=="native_white":
                    np.save(HERE/"native_white_test_pred.npy",pred)
    write("projection_spectrum.csv",condition)
    write("feature_spectrum.csv",features)
    write("probes.csv",probes)
    write("lambda_grid.csv",candidates)


def hybrids(meta):
    arch=torch.load(EXP/"mmimdb_joint_residual_BO9efZ/test_predictions.pt",map_location="cpu")
    ds=meta["test"]
    rows=ds["rows"]
    y=ds["labels"][rows]
    assert np.array_equal(rows,arch["rows"].numpy()) and np.array_equal(y,arch["y"].numpy())
    start=torch.load(MAIN/"k1/test_epoch_11.pt",map_location="cpu").numpy()>0
    assert start.shape==(len(rows),3,23)
    with np.load(MATCH/"dpl.npz") as f:
        dpl=f["test_pred"][rows]
        dpl_full=f["complete_pred"]
    heads=torch.load(SWAP/"shared_heads.pt",map_location="cpu")
    with np.load(SWAP/"shared_pred.npz") as f:
        probe={k:(f["test_"+k].astype(float)>np.asarray(heads[k]["cuts"]))[rows] for k in ("raw","native")}
    probe["white"]=np.load(HERE/"native_white_test_pred.npy")[rows]
    source=read(MAIN/"per_seed.csv")
    records,aggregate,decomposition=[],[],[]
    for direction in ("text","both","image"):
        for rate in (50,70,90):
            results=[]
            for seed in SEEDS:
                mask=arch["masks"][direction,rate,seed].numpy()[rows]
                idx=np.arange(len(rows))
                current=start[idx,mask]
                prev=next(r for r in source if r["mode"]==direction and int(r["rate"])==rate and int(r["seed"])==seed)
                def metric(p):
                    return fscore(counts(y,p,np.ones(len(y),int)).sum(0))
                score=metric(current)
                assert abs(score-float(prev["off_text90"]))<1e-6
                result=dict(mode=direction,rate=rate,seed=seed,main=score,dpl_expert=float(prev["DPL"]))
                for name,replacement in dict(dpl90_poster=dpl,raw_probe=probe["raw"],native_probe=probe["native"],
                                             whitened_probe=probe["white"]).items():
                    changed=current.copy()
                    changed[mask==1]=replacement[mask==1]
                    result[name]=metric(changed)
                if direction=="text" and rate==90:
                    assert np.array_equal(ds["masks"][SEEDS.index(seed)][rows],mask)
                    ix=ds["complete_indices"]
                    assert np.isin(idx[mask==0],ix).all()
                    full=current.copy()
                    full[ix[mask[ix]==0]]=dpl_full[mask[ix]==0]
                    full_only=metric(full)
                    full[mask==1]=dpl[mask==1]
                    both=metric(full)
                    assert abs(both-result["dpl_expert"])<.001
                    post=result["dpl90_poster"]
                    decomposition.append(dict(seed=seed,main=score,swap_poster=post,swap_complete=full_only,
                        swap_both=both,poster_shapley=((post-score)+(both-full_only))/2,
                        complete_shapley=((full_only-score)+(both-post))/2))
                records.append(result)
                results.append(result)
            aggregate.append(dict(mode=direction,rate=rate,**{k:np.mean([r[k] for r in results])
                    for k in results[0] if k not in ("mode","rate","seed")}))
    write("per_mask.csv",records)
    write("hybrids.csv",aggregate)
    write("decomposition.csv",decomposition)


if __name__=="__main__":
    torch.set_num_threads(2)
    meta=metadata()
    spectrum_and_probes(meta)
    hybrids(meta)
