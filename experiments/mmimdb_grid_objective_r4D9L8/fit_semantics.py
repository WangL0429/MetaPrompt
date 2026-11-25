"""Full, paired 20-epoch second-order training of the conditional semantic generator."""
import argparse
import json
import math
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from torch.nn import functional as F
from predictor import SemanticPredictor,paired_nll
from semantic_io import (JOB,SOURCE,setup,make_schedule,choose_weights,val_grid,store,json_out)

EPOCHS,STEPS=20,64
MODES=("text","image","both")


def read(split):
    d=torch.load(JOB/f"features_{split}.pt",map_location="cpu")
    return {k:v.cuda() for k,v in d.items()}


def get_weights():
    head=torch.load(SOURCE,map_location="cpu")["state_dict"]["mmimdb_classifier.weight"].cuda().float()
    return {1:head[:,512:],2:head[:,:512]}


def run_gen(net,params,d,idx,state,w):
    args=(d[f"h{state}"][idx],d[f"x{state}"][idx],d[f"mask{state}"][idx],state,d["base"][idx,state],w[state])
    if params is None:return net(*args)
    return functional_call(net,params,args)


def masks(size,rate,rng):
    order=rng.permutation(size)
    count=round(rate*size)
    half=(count+int(rng.integers(2)))//2
    result=[]
    for take in (count,0,half):
        mode=np.zeros(size,dtype=np.int64)
        mode[order[:take]]=1
        mode[order[take:count]]=2
        result.append(torch.as_tensor(mode,device="cuda"))
    return result


def loss(net,params,d,ids,mode,w,stats,pos,paired,outer,*,semantic_weight=.35,grid_terms=None):
    ix=torch.as_tensor(ids,device="cuda")
    target=d["y"][ix].float()
    reference=d["base"][ix,mode]
    z=reference.clone()
    semantic=reference.new_zeros(())
    for state in (1,2):
        use=(mode==state) | ((mode==0) if paired else torch.zeros_like(mode,dtype=torch.bool))
        selected=use.nonzero().flatten()
        if not len(selected):continue
        out,aux=run_gen(net,params,d,ix[selected],state,w)
        same=mode[selected]==state
        z=z.index_copy(0,selected[same],out[same])
        complete=~same
        if paired and semantic_weight and complete.any():
            paired_ix=ix[selected[complete]]
            other=d["complete"][paired_ix].chunk(2,1)[2-state]
            semantic=semantic+.5*complete.sum()/len(ix)*paired_nll(tuple(a[complete] for a in aux),other)
    if outer:
        fraction,denom=stats
        dp=(z/.5).sigmoid()-(reference/.5).sigmoid()
        if grid_terms is None:
            risk=(-2*(target-fraction)*dp/denom).mean()
        else:
            slope,intercept=grid_terms
            risk=((intercept[mode]-target*slope[mode])*dp).mean()
        classification=F.binary_cross_entropy_with_logits(z,target,pos_weight=pos)
        classification=classification+.5*risk+.02*(z-reference).square().mean()
    else:
        classification=F.binary_cross_entropy_with_logits(z,target)
    return classification+semantic_weight*semantic


def support_router(rows,groups,batch,rng):
    used=set(rows[batch].tolist())
    extra=[]
    while len(extra)<40:
        movie=int(rng.integers(len(groups)))
        if movie not in used:
            used.add(movie)
            extra.append(int(rng.choice(groups[movie])))
    return extra[:16],extra[16:]


@torch.no_grad()
def predictions(net,d,w):
    z=d["base"].clone()
    stats={}
    for state in (1,2):
        sums=z.new_zeros(4)
        for idx in torch.arange(len(z),device=z.device).split(256):
            scores,(direction,magnitude,mix,gate)=run_gen(net,None,d,idx,state,w)
            z[idx,state]=scores
            pi=mix.exp()
            alignment=(direction@direction.transpose(1,2))
            separation=(1-alignment).sum((1,2))/max(1,net.k*(net.k-1))
            sums+=torch.stack((gate.sigmoid().sum(),(-pi*mix).sum(),
                               separation.sum(),(scores-d["base"][idx,state]).abs().mean(1).sum()))
        stats[str(state)]=(sums/len(z)).tolist()
    return z.cpu(),stats


def snapshot(net):
    return {k:p.detach().cpu().clone() for k,p in net.state_dict().items()}


def fit(k, *, paired_supervision=True, out=None, warm_start=None, grid_objective=False, lr=4e-4):
    setup()
    out=JOB/f"k{k}" if out is None else out
    out.mkdir(exist_ok=True)
    semantic_weight = .35 if paired_supervision else 0.
    net=SemanticPredictor(k).cuda()
    initialization = None
    if warm_start is not None:
        checkpoint = torch.load(warm_start,map_location="cpu")
        spec = checkpoint["semantic_predictor"]
        assert spec["k"] == k
        net.load_state_dict(spec["weights"],strict=True)
        initialization = str(warm_start.resolve())
        del checkpoint,spec
    recipe = dict(initialization=initialization,grid_objective=grid_objective,lr=lr)
    w=get_weights()
    d,dev=read("train"),read("dev")
    cal=dev["cal"].cpu()
    labels=dev["y"].cpu()
    assert len(d["y"])==32278 and dev["rows"][dev["cal"]].unique().numel()==1444
    population=d["y"].float().mean(0)
    pos=((1-population)/population.clamp_min(1e-6)).sqrt().clamp(1,8)
    p=(d["base"]/.5).sigmoid()
    tp=(p*d["y"][:,None]).mean(0)
    pred=p.mean(0)
    rows=d["rows"].cpu().numpy()
    groups=[[] for _ in range(int(rows.max())+1)]
    for i,r in enumerate(rows):groups[r].append(i)
    schedule=make_schedule(STEPS*EPOCHS,0,"curriculum")
    named={n:p for n,p in net.named_parameters() if p.requires_grad}
    params=tuple(named.values())
    opt=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
    base=val_grid(dev["base"].transpose(0,1).cpu()[:,cal],labels[cal])
    history=[]
    if (out/"resume.pt").exists():
        prev=torch.load(out/"resume.pt",map_location="cpu")
        if prev.get("semantic_weight",.35) != semantic_weight:
            raise ValueError("The checkpoint uses a different paired-supervision weight")
        if prev.get("recipe",dict(initialization=None,grid_objective=False,lr=4e-4)) != recipe:
            raise ValueError("The checkpoint uses a different objective or initialization")
        net.load_state_dict(prev["net"]); opt.load_state_dict(prev["opt"])
        history=prev["history"]
    for epoch in range(len(history)+1,EPOCHS+1):
        tic=time.monotonic()
        agg=np.zeros(6)
        counts=np.zeros(3,dtype=int)
        grid_terms=None
        if grid_objective:
            from grid_objective import grid_coefficients
            logits,_=predictions(net,d,w)
            grid_terms=grid_coefficients((logits.to(population.device)/.5).sigmoid(),d["y"].float())
            del logits
        perm=np.random.default_rng(1000+epoch).permutation(len(rows))
        for b,batch in enumerate(np.array_split(perm,STEPS)):
            step=(epoch-1)*STEPS+b
            rate,_,seed=schedule[step]
            rng=np.random.default_rng([seed,0])
            support,router=support_router(rows,groups,batch,rng)
            sm=masks(len(support),rate,rng); rm=masks(len(router),rate,rng)
            mix=tp.new_tensor([1-rate,rate/2,rate/2])
            den=(mix@pred+population).clamp_min(1e-5)
            stats=((mix@tp)/den,den)
            directions=[]
            for mask in sm:
                sup=loss(net,None,d,support,mask,w,stats,pos,True,False,semantic_weight=semantic_weight)
                gs=torch.autograd.grad(sup,params,create_graph=True,allow_unused=True)
                gs=[torch.zeros_like(p) if g is None else g for g,p in zip(gs,params)]
                norm=torch.cat([g.reshape(-1) for g in gs]).norm().clamp_min(1e-8)
                directions.append([g/norm for g in gs])
            with torch.no_grad():
                def route(candidate):
                    return torch.stack([loss(net,candidate,d,router,m,w,stats,pos,False,True) for m in rm])
                ref=route(None)
                custom=[]
                for dr in directions:
                    cand={n:p-.03*g for (n,p),g in zip(named.items(),dr)}
                    custom.append(route(cand)-ref)
                weights=choose_weights(torch.stack(custom).cpu().numpy()).tolist()
            fast={n:p-.03*sum(wi*dr[i] for wi,dr in zip(weights,directions))
                  for i,(n,p) in enumerate(named.items())}
            total=0.
            for side,part in enumerate(np.array_split(batch,2)):
                mode=masks(len(part),rate,rng)[2]
                counts+=torch.bincount(mode,minlength=3).cpu().numpy()
                total+=len(part)/len(batch)*loss(net,None if side==0 else fast,d,part,mode,w,stats,pos,True,True,
                                                 semantic_weight=semantic_weight,grid_terms=grid_terms)
            opt.zero_grad(set_to_none=True)
            total.backward()
            norm=nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=lr*(.2+.4*(1+math.cos(math.pi*step/(STEPS*EPOCHS-1))))
            opt.step()
            agg+=np.array([total.item(),norm.item(),rate,*weights])
        assert counts.sum()==len(rows)
        z,diagnostics=predictions(net,dev,w)
        grid=val_grid(z.transpose(0,1)[:,cal],labels[cal])
        assert np.isfinite(grid).all()
        item=dict(epoch=epoch,cal_grid=grid,seconds=time.monotonic()-tic,
                  train_plots=len(rows),query_state_counts=counts.tolist(),mean_training=(agg/STEPS).tolist(),
                  gate_entropy_separation_abslogit=diagnostics)
        history.append(item)
        store(out/f"epoch_{epoch:02}.pt",snapshot(net))
        store(out/"resume.pt",dict(net=snapshot(net),opt=opt.state_dict(),history=history,semantic_weight=semantic_weight,recipe=recipe))
        if epoch==1:json_out(out/"first_epoch.json",item)
        print(k,epoch,round(float(np.mean(grid)),4),round(grid[-1],4),flush=True)
    picks={"mean9":max(history,key=lambda r:(np.mean(r["cal_grid"]),r["cal_grid"][-1]))["epoch"],
           "text90":max(history,key=lambda r:(r["cal_grid"][-1],np.mean(r["cal_grid"])))["epoch"]}
    for epoch in set(picks.values()):
        ck=torch.load(SOURCE,map_location="cpu")
        ck["semantic_predictor"]=dict(k=k,epoch=epoch,semantic_weight=semantic_weight,recipe=recipe,
                                      weights=torch.load(out/f"epoch_{epoch:02}.pt",map_location="cpu"))
        store(out/f"deploy_epoch{epoch}.ckpt",ck)
        del ck
    for selector,epoch in picks.items():
        path=out/(selector+".ckpt")
        if not path.exists():path.symlink_to(out/f"deploy_epoch{epoch}.ckpt")
    json_out(out/"selection.json",dict(k=k,picks=picks,epochs=EPOCHS,steps=STEPS,seed=0,history=history,
                                      semantic_weight=semantic_weight,recipe=recipe,native_cal=base,trainable=sum(p.numel() for p in params)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("k",type=int)
    parser.add_argument("--no-paired-supervision",action="store_true")
    args=parser.parse_args()
    fit(args.k,paired_supervision=not args.no_paired_supervision)
