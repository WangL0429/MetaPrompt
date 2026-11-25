"""Package and replay the archived 768-dimensional linear probe with its locked calibration thresholds."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diagnose import HERE, EXP, MATCH, MAIN, SWAP, SEEDS, metadata, read, write

sys.path.insert(0,str(EXP/"mmimdb_pair_ablation_dZfki9"))
from predictor import load_predictor_checkpoint, infer
from semantic_io import plots, setup


def package(path):
    ck=torch.load(MAIN/"k1/text90.ckpt",map_location="cpu")
    probe=torch.load(SWAP/"shared_heads.pt",map_location="cpu")["raw"]
    ck["poster_readout"]=dict(weight=probe["w"],bias=probe["b"],
        thresholds=torch.as_tensor(probe["cuts"]),ridge=probe["ridge"],input="image_encoder.ln_post")
    assert ck["poster_readout"]["weight"].shape==(23,768)
    tmp=path.with_suffix(".part")
    torch.save(ck,tmp)
    tmp.replace(path)


def load_readout(path):
    model,gen=load_predictor_checkpoint(path)
    head=torch.load(path,map_location="cpu")["poster_readout"]
    return model,gen,{k:head[k].cuda() for k in ("weight","bias","thresholds")}


@torch.inference_mode()
def forward(net,gen,head,batch):
    modes=torch.as_tensor(batch["missing_type"],device="cuda")
    active=modes==1
    if not active.any():
        return infer(net,gen,batch)
    raw=[]
    def capture(module,inputs,output):
        if output.ndim==2:
            raw.append(output)
    hook=net.model.image_encoder.ln_post.register_forward_hook(capture)
    try:
        z=infer(net,gen,batch)
    finally:
        hook.remove()
    assert len(raw)==1 and len(raw[0])==len(modes)
    logits=F.linear(raw[0][active].half().float(),head["weight"],head["bias"])
    z[active]=(logits.double()-head["thresholds"]).float()
    return z


def score_counts(y,p):
    tp=np.logical_and(y,p).sum(0)
    return 200*(tp/np.maximum(y.sum(0)+p.sum(0),1)).mean()


def paired_intervals(y, rows, predictions):
    tables=[]
    names=list(predictions)
    n=rows.max()+1
    for name in names:
        p=predictions[name]
        c=np.stack((y[:,None]*p,y[:,None].astype(np.int16)+p),axis=2)
        t=np.zeros((n,*c.shape[1:]),dtype=float)
        np.add.at(t,rows,c)
        tables.append(t.reshape(n,-1))
    tables=np.concatenate(tables,axis=1)
    def score(mult):
        totals=(mult@tables).reshape(len(mult),len(names),3,2,23)
        return 200*(totals[:,:,:,0]/np.maximum(totals[:,:,:,1],1)).mean((2,3))
    rng=np.random.default_rng(1027)
    point=score(np.ones((1,n)))[0]
    draws=np.concatenate([score(rng.multinomial(n,np.full(n,1/n),size=50).astype(float)) for _ in range(30)])
    result=[]
    for other in ("main","dpl"):
        a,b=names.index("replay"),names.index(other)
        lo,hi=np.percentile(draws[:,a]-draws[:,b],[2.5,97.5])
        result.append(dict(metric="text90_macro_F1",reference=other,delta=point[a]-point[b],
                           lower95=lo,upper95=hi,draws=len(draws),movies=n))
    return result


def main():
    setup()
    path=HERE/"raw_readout_diagnostic.ckpt"
    package(path)
    net,gen,head=load_readout(path)
    ds=plots(net.hparams.config,"test")
    m=metadata()["test"]
    rows=m["rows"]
    assert [str(x).zfill(7) for x in ds.table["image_id"].to_pylist()]==m["ids"]
    assert np.array_equal([ds.index_mapper[i][0] for i in range(len(ds))],rows)
    assert np.array_equal(ds.table["label"].to_pylist(),m["labels"])
    y=m["labels"][rows]
    base=torch.load(MAIN/"k1/test_epoch_11.pt",map_location="cpu").numpy()>0
    with np.load(SWAP/"shared_pred.npz") as cache:
        reference=(cache["test_raw"].astype(float)>head["thresholds"].cpu().numpy())[rows]
    active_scores=[]
    mixed_diff=0
    for step,(image,texts) in enumerate(DataLoader(ds,batch_size=64,num_workers=5,pin_memory=True)):
        image=image.cuda()
        batch_out=[]
        for state in ((0,1,2,3) if step==0 else (0,1,2)):
            modes=np.arange(len(image))%3 if state==3 else np.full(len(image),state)
            missing=torch.from_numpy(modes==2).cuda()[:,None,None,None]
            batch=dict(image=[image.masked_fill(missing,0)],text=(["" if s==1 else t for s,t in zip(modes,texts)],),
                       missing_type=modes.tolist())
            p=forward(net,gen,head,batch).cpu().numpy()>0
            if state==3:
                mixed_diff=int((np.stack(batch_out,axis=1)[np.arange(len(image)),modes]!=p).sum())
            else:
                batch_out.append(p)
        active_scores.append(np.stack(batch_out,1))
    all_predictions=np.concatenate(active_scores)
    differences=[int((all_predictions[:,s]!=r).sum()) for s,r in enumerate((base[:,0],reference,base[:,2]))]
    arch=torch.load(EXP/"mmimdb_joint_residual_BO9efZ/test_predictions.pt",map_location="cpu")
    diag={(r["mode"],int(r["rate"])):r for r in read(HERE/"hybrids.csv")}
    evidence=[]
    high={name:[] for name in ("replay","dpl","main")}
    with np.load(MATCH/"dpl.npz") as f:
        poster=f["test_pred"][rows]
        full=f["complete_pred"]
    for direction in ("text","both","image"):
        for rate in (50,70,90):
            scores=[]
            for seed in SEEDS:
                mode=arch["masks"][direction,rate,seed].numpy()[rows]
                idx=np.arange(len(rows))
                pred=all_predictions[idx,mode]
                scores.append(score_counts(y,pred))
                if (direction,rate)==("text",90):
                    spec=poster.copy()
                    ix=m["complete_indices"]
                    spec[ix[mode[ix]==0]]=full[mode[ix]==0]
                    for k,v in (("replay",pred),("dpl",spec),("main",base[idx,mode])):
                        high[k].append(v)
            ref=diag[direction,rate]
            evidence.append(dict(mode=direction,rate=rate,native=np.mean(scores),offline=float(ref["raw_probe"]),
                DPL=float(ref["dpl_expert"]),gain_vs_DPL=np.mean(scores)-float(ref["dpl_expert"])))
    write("raw_replay.csv",evidence)
    write("paired_text90.csv",paired_intervals(y,rows,{k:np.stack(v,1) for k,v in high.items()}))
    (HERE/"replay_checks.json").write_text(json.dumps(dict(items=len(rows),movies=len(m["ids"]),
        decisions_per_state=int(y.size),state_mismatches=differences,mixed_batch_mismatches=mixed_diff,
        max_abs_grid_diff=max(abs(r["native"]-r["offline"]) for r in evidence)),indent=2))


if __name__=="__main__":main()
