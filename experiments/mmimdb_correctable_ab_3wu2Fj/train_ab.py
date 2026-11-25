"""Matched two-arm experiment using the existing three-direction second-order prompt update."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
BASE=EXP/"mmimdb_joint_full_9aNxqw"
SOURCE=EXP/"mmimdb_joint_head_UR2YBk/selected.ckpt"
sys.path.insert(0,str(BASE))
import train as meta
from model_io import load,plots,outputs,views,validate
from common import split_movies
from rule import flat,unit,curvature

EPOCHS=20
STEPS=64
PAIR_WEIGHT=.1
MICRO=int(os.environ.get("PAIR_MICRO","16"))


def snapshot(learner):
    return {k:p.detach().cpu().clone() for k,p in learner.named_parameters()}


def save(obj,path):
    tmp=path.with_suffix(".part")
    torch.save(obj,tmp)
    tmp.replace(path)


def write_json(path,value):
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(value,indent=2))
    tmp.replace(path)


def get_data():
    data=meta.TrainData()
    bank=torch.load(JOB/"training_pairs.pt",map_location="cpu")
    assert np.array_equal(bank["rows"],data.rows)
    data.ref=bank["reference"].cuda()
    y=data.labels[data.rows]
    p=(data.ref/meta.T).sigmoid()
    data.tp=(p*y).mean(1)
    data.pred=p.mean(1)
    return data,bank["banks"],bank["oof"]


def calibration(net,ck):
    ds=plots(ck["hyper_parameters"]["config"],"dev")
    ids=[str(x).zfill(7) for x in ds.table["image_id"].to_pylist()]
    rows=np.asarray([ds.index_mapper[i][0] for i in range(len(ds))])
    mask=split_movies(ids)
    assert mask.sum()==1444 and len(mask)==2608
    ix=np.flatnonzero(mask.numpy()[rows])
    y=torch.tensor(ds.table["label"].to_pylist())[rows[ix]]
    return DataLoader(Subset(ds,ix.tolist()),batch_size=64,num_workers=3,pin_memory=True),y


def matched_rows(banks,state,n,rng,excluded,rows,digest,stats,oof,ref,arm):
    selected=[]
    targets=[]
    for _ in range(n):
        c=int(rng.integers(len(banks[state])))
        bank=banks[state][c]
        for attempt in range(1000):
            k=int(rng.integers(len(bank)))
            a,an,b,bn=map(int,bank[k])
            if excluded.isdisjoint(rows[[a,an,b,bn]]):
                break
        else:raise RuntimeError("No movie-disjoint matched slot")
        excluded.update(rows[[a,an,b,bn]])
        digest.update(np.array([state,c,k],dtype="<i4").tobytes())
        da=float(ref[state,an,c]-ref[state,a,c])
        db=float(ref[state,bn,c]-ref[state,b,c])
        assert min(da,db)>0 and abs(da-db)<=.02001
        margins=oof[state]
        stats[state-1]+=np.array([1,da,db,margins[a,c]-margins[an,c]>=.25,
                                   margins[b,c]-margins[bn,c]>=.25])
        selected.extend((a,an) if arm=="A" else (b,bn))
        targets.append(c)
    return np.asarray(selected),torch.tensor(targets,device="cuda")


def joint(net,names,params,data,banks,oof,ref,global_ix,idx,rate,seed,shard,arm,digest,stats):
    rng=np.random.default_rng([seed,shard])
    pair_rng=np.random.default_rng([seed,shard,7711])
    support,router=data.extra(rng,global_ix)
    excluded=set(data.rows[np.r_[global_ix,support,router]].tolist())
    sup,route=data.batch(support),data.batch(router)
    sm,rm=meta.masks(len(support),rate,rng),meta.masks(len(router),rate,rng)
    population=data.population_stats(rate)
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
        dirs=torch.stack([unit(flat(meta.support_loss(net,sup,m),params)) for m in sm])
    baseline=meta.routing(net,data,router,rm,None,route,population)
    changes=[]
    for direction in dirs:
        with torch.no_grad():
            cand={n:p-d for n,p,d in zip(names,params,views(meta.ALPHA*direction,params))}
        changes.append(meta.routing(net,data,router,rm,cand,route,population)-baseline)
    w=dirs.new_tensor(meta.choose_weights(torch.stack(changes).cpu().numpy()))
    fast={n:(p-d).detach().requires_grad_() for n,p,d in zip(names,params,views(meta.ALPHA*(w@dirs),params))}
    live_grad,fast_grad=torch.zeros_like(dirs[0]),torch.zeros_like(dirs[0])
    for side,items in enumerate(np.array_split(idx,2)):
        dest,parameters,cand=(live_grad,params,None) if side==0 else (fast_grad,tuple(fast.values()),fast)
        mode=np.zeros(len(items),dtype=int)
        perm=rng.permutation(len(items))
        count=round(rate*len(items))
        half=(count+int(rng.integers(2)))//2
        mode[perm[:half]]=1
        mode[perm[half:count]]=2
        for state in (0,1,2):
            members=items[mode==state]
            for j in range(0,len(members),MICRO):
                sub=members[j:j+MICRO]
                img,tok,y=data.batch(sub)
                with torch.autocast("cuda",dtype=torch.float16):
                    z=outputs(net,img,tok,state,cand)[0]
                loss=meta.risk(z,y,data.ref[state,sub],data.pos,population)
                dest.add_(flat(128*loss,parameters),alpha=len(sub)/(128*len(idx)))
            if state and len(members)>=2:
                pairs=min(4,max(1,round(len(members)/8)))
                extra,c=matched_rows(banks,state,pairs,pair_rng,excluded,data.rows,digest,stats,oof,ref,arm)
                img,tok,y=data.batch(extra)
                with torch.autocast("cuda",dtype=torch.float16):
                    z=outputs(net,img,tok,state,cand)[0]
                j=torch.arange(0,len(extra),2,device=z.device)
                assert (y[j,c]==1).all() and (y[j+1,c]==0).all()
                loss=F.softplus(z[j+1,c]-z[j,c]).mean()
                dest.add_(flat(128*loss,parameters),alpha=PAIR_WEIGHT*len(members)/(128*len(idx)))
    correction=torch.zeros_like(live_grad)
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
        for wi,m in zip(w.tolist(),sm):
            correction.add_(curvature(meta.support_loss(net,sup,m),params,fast_grad),alpha=meta.ALPHA*wi)
    grad=live_grad+fast_grad-correction
    assert grad.isfinite().all()
    return grad,float(correction.norm()/fast_grad.norm().clamp_min(1e-12))


def main(arm):
    assert arm in ("A","B")
    out=JOB/arm
    out.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    net,ck=load(SOURCE)
    net.cuda().eval().requires_grad_(False)
    learner=net.model.prompt_learner.requires_grad_(True)
    names,params=zip(*learner.named_parameters())
    data,banks,oof=get_data()
    ref=data.ref.cpu().numpy()
    val,y=calibration(net,ck)
    plan=meta.make_schedule(EPOCHS*STEPS,0,"curriculum")
    opt=torch.optim.AdamW(params,lr=meta.LR,weight_decay=0)
    resume=out/"resume.pt"
    if resume.exists():
        stored=torch.load(resume,map_location="cpu")
        with torch.no_grad():
            for name,p in learner.named_parameters():
                p.copy_(stored["current"][name])
        opt.load_state_dict(stored["optimizer"])
        history,best,trained=stored["history"],stored["best"],stored["trained"]
        base=history[0]["grid"]
    else:
        base=validate(net,val,y)
        history=[dict(epoch=0,grid=base,eligible=True)]
        best=(0,(base[-1],sum(base)),snapshot(learner))
        trained=None
    for epoch in range(len(history),EPOCHS+1):
        tic=time.monotonic()
        digest=hashlib.sha256()
        stats=np.zeros((2,5))
        ratios=[]
        order=np.random.default_rng(1000+epoch).permutation(len(data.rows))
        for b,global_ix in enumerate(np.array_split(order,STEPS)):
            step=(epoch-1)*STEPS+b
            rate,_,seed=plan[step]
            total=None
            for shard,idx in enumerate(np.array_split(global_ix,2)):
                g,c=joint(net,names,params,data,banks,oof,ref,global_ix,idx,rate,seed,shard,arm,digest,stats)
                ratios.append(c)
                g.mul_(len(idx)/len(global_ix))
                total=g if total is None else total.add_(g)
            opt.param_groups[0]["lr"]=meta.LR*(.2+.4*(1+math.cos(math.pi*step/(len(plan)-1))))
            opt.zero_grad(set_to_none=True)
            for p,g in zip(params,views(total,params)):p.grad=g
            opt.step()
            if step==0:
                write_json(out/"startup.json",dict(step=1,gradient_norm=total.norm().item(),relative_curvature=ratios,
                                                   trainable=sum(p.numel() for p in params),slots=stats.tolist()))
        grid=validate(net,val,y)
        ok=meta.accepted(grid,base) and sum(grid)>=sum(base)
        rank=(grid[-1],sum(grid))
        if ok and rank>best[1]:best=(epoch,rank,snapshot(learner))
        if trained is None or (ok,*rank)>trained[1]:trained=(epoch,(ok,*rank),snapshot(learner))
        assert np.array_equal(stats[:,0],stats[:,4]) and (stats[:,0]>0).all()
        history.append(dict(epoch=epoch,grid=grid,eligible=ok,slots_sha256=digest.hexdigest(),
                            paired_stats=stats.tolist(),relative_curvature=float(np.mean(ratios)),
                            seconds=time.monotonic()-tic))
        save(dict(current=snapshot(learner),optimizer=opt.state_dict(),history=history,best=best,trained=trained),resume)
        if epoch==1:write_json(out/"first_epoch.json",history[-1])
        print(arm,epoch,grid[-1],ok,flush=True)
    for label,choice in (("selected",best),("trained",trained)):
        epoch,_,weights=choice
        path=out/(label+".ckpt")
        if path.exists():continue
        if epoch==0:path.symlink_to(SOURCE)
        elif label=="trained" and epoch==best[0]:path.symlink_to(out/"selected.ckpt")
        else:
            for n,t in weights.items():ck["state_dict"]["model.prompt_learner."+n]=t
            ck["correctable_pair_ab"]=dict(arm=arm,epoch=epoch,seed=0,source=str(SOURCE))
            save(ck,path)
    write_json(out/"selection.json",dict(arm=arm,seed=0,epochs=EPOCHS,steps=STEPS,
               selected=best[0],trained=trained[0],history=history,source=str(SOURCE),
               trainable=sum(p.numel() for p in params),pair_weight=PAIR_WEIGHT))


if __name__=="__main__":main(sys.argv[1])
