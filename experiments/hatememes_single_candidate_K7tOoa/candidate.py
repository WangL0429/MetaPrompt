import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from single_label import N_CLASSES, PRIMARY, TAG, TASK, decision, head as classifier, measure
from food_io import (SingleLabelDataset, JOB, condition_masks, csv_out, encode, extra_indices, grid, load_native,
                     make_schedule, mix_masks, save, setup)
from meta_weights import choose_weights
from predictor import SemanticPredictor

EPOCHS, BATCH = 20,512


def candidate_scores(gen, fast, d, ix, s, w):
    args = (d[f"h{s}"][ix],d[f"x{s}"][ix],d[f"mask{s}"][ix],s,d[f"base{s}"][ix],w[s])
    return (gen(*args) if fast is None else functional_call(gen,fast,args))[0]


def loss(gen, fast, d, ix, mask, w, anchor):
    mode=torch.as_tensor(mask,device="cuda")
    # A zero-missing support can have a constant objective.
    total=next(iter(gen.parameters())).reshape(-1)[0]*0
    for state in range(3):
        selected=ix[mode==state]
        if len(selected)==0:
            continue
        ref=d[f"base{state}"][selected]
        z=ref if state==0 else candidate_scores(gen,fast,d,selected,state,w)
        val=F.cross_entropy(z,d["y"][selected])
        if anchor and state:
            val=val+.02*F.kl_div(z.log_softmax(-1),ref.softmax(-1),reduction="batchmean")
        total=total+len(selected)/len(ix)*val
    return total


def update(gen,named,opt,d,batch,rate,seed,w):
    rng=np.random.default_rng(seed)
    extra=extra_indices(len(d["y"]),batch,40,rng)
    support,router=torch.as_tensor(extra[:16],device="cuda"),torch.as_tensor(extra[16:],device="cuda")
    sm,rm=mix_masks(16,rate,rng),mix_masks(24,rate,rng)
    params=tuple(named.values())
    directions=[]
    for m in sm:
        gs=torch.autograd.grad(loss(gen,None,d,support,m,w,False),params,create_graph=True,allow_unused=True)
        gs=[torch.zeros_like(p) if g is None else g for p,g in zip(params,gs)]
        norm=torch.cat([g.reshape(-1) for g in gs]).norm().clamp_min(1e-12)
        directions.append([g/norm for g in gs])
    with torch.no_grad():
        def route(fast):
            return torch.stack([loss(gen,fast,d,router,m,w,True) for m in rm])
        base=route(None)
        deltas=[]
        for dr in directions:
            fast={k:p-.03*g for (k,p),g in zip(named.items(),dr)}
            deltas.append(route(fast)-base)
        weights=choose_weights(torch.stack(deltas).cpu().numpy()).tolist()
    fast={k:p-.03*sum(a*dr[j] for a,dr in zip(weights,directions)) for j,(k,p) in enumerate(named.items())}
    total=0
    for side,part in enumerate(np.array_split(batch,2)):
        mask=mix_masks(len(part),rate,rng)[2]
        total+=len(part)/len(batch)*loss(gen,fast if side else None,d,torch.as_tensor(part,device="cuda"),mask,w,True)
    opt.zero_grad(set_to_none=True)
    total.backward()
    torch.nn.utils.clip_grad_norm_(params,2,error_if_nonfinite=True)
    opt.step()


@torch.no_grad()
def predict(gen,d,w):
    parts=[decision(d["base0"])]
    for s in (1,2):
        parts.append(torch.cat([decision(candidate_scores(gen,None,d,i,s,w))
                                for i in torch.arange(len(d["y"]),device=w[1].device).split(256)]))
    return torch.stack(parts,1).cpu()


def fit_raw(train,val):
    x=train["raw"].float()
    mean=x.mean(0)
    std=x.std(0,unbiased=False).clamp_min(1e-5)
    x=(x-mean)/std
    candidates=[]
    records=[]
    for ridge in (.0003,.003,.03,.3,3.,30.,300.):
        w=torch.zeros(N_CLASSES,x.size(1),device=x.device,requires_grad=True)
        b=torch.zeros(N_CLASSES,device=x.device,requires_grad=True)
        opt=torch.optim.LBFGS([w,b],lr=1.,max_iter=220,tolerance_grad=1e-6,line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            l=F.cross_entropy(F.linear(x,w,b),train["y"])+.5*ridge*w.square().sum()/len(b)
            l.backward()
            return l
        opt.step(closure)
        with torch.no_grad():
            coeff=w/std
            bias=b-coeff@mean
            z=F.linear(val["raw"].float(),coeff,bias)
            score=measure(decision(z),val["y"])
            nll=F.cross_entropy(z,val["y"]).item()
            assert math.isfinite(nll)
            candidates.append((score,-nll,ridge,{"weight":coeff.cpu(),"bias":bias.cpu(),"ridge":ridge}))
            records.append({"ridge":ridge,f"val_image_only_{PRIMARY}":score,"val_nll":nll})
    csv_out(JOB/"readout_validation.csv",records)
    return max(candidates,key=lambda r:r[:3])[-1]


def main():
    setup()
    net=load_native(JOB/"prompt_selected.ckpt")
    splits={s:SingleLabelDataset(s) for s in ("train","val")}
    masks=condition_masks(splits["val"].ids,"val")
    d={}
    for s,ds in splits.items():
        path=JOB/f"{s}_features.pt"
        if not path.exists():
            save(encode(net,ds,features=True),path)
        data=torch.load(path,map_location="cpu")
        assert torch.equal(data["y"],ds.labels)
        d[s]={k:v.cuda() for k,v in data.items()}
    w={1:classifier(net).weight[:,512:].detach(),2:classifier(net).weight[:,:512].detach()}
    gen=SemanticPredictor(1,categorical=True).cuda()
    named={k:p for k,p in gen.named_parameters() if p.requires_grad}
    opt=torch.optim.AdamW(named.values(),lr=4e-4,weight_decay=.01)
    n=len(splits["train"])
    steps=math.ceil(n/BATCH)
    sched=make_schedule(EPOCHS*steps,0,"curriculum")
    best,best_epoch,best_state=None,None,None
    history=[]
    resume=JOB/"candidate_resume.pt"
    if resume.exists():
        r=torch.load(resume,map_location="cpu")
        gen.load_state_dict(r["current"])
        opt.load_state_dict(r["opt"])
        best,best_epoch,best_state,history=r["best"],r["epoch"],r["best_state"],r["history"]
    def snapshot():
        return {k:p.detach().cpu().clone() for k,p in gen.state_dict().items()}
    for ep in range(len(history)+1,EPOCHS+1):
        perm=np.random.default_rng(1000+ep).permutation(n)
        for b,part in enumerate(np.array_split(perm,steps)):
            i=(ep-1)*steps+b
            rate,_,seed=sched[i]
            opt.param_groups[0]["lr"]=4e-4*(.6+.4*math.cos(math.pi*i/(len(sched)-1)))
            update(gen,named,opt,d["train"],part,rate,seed,w)
        scores,_=grid(predict(gen,d["val"],w),d["val"]["y"],masks)
        key=(scores["text_90"],float(np.mean(list(scores.values()))))
        if best is None or key>best:
            best,best_epoch,best_state=key,ep,snapshot()
        history.append({"epoch":ep,PRIMARY:scores})
        save(dict(current=snapshot(),opt=opt.state_dict(),best=best,epoch=best_epoch,best_state=best_state,
                  history=history),resume)
        print(ep,scores["text_90"],best_epoch,flush=True)
    head=fit_raw(d["train"],d["val"])
    ck=torch.load(JOB/"prompt_selected.ckpt",map_location="cpu")
    ck[f"{TAG}_candidate"]={"epoch":best_epoch,"weights":best_state,"k":1,"categorical":True}
    ck[f"{TAG}_image_readout"]=head
    save(ck,JOB/f"{TASK}_selected.ckpt")
    (JOB/"selection.json").write_text(json.dumps(dict(seed=0,criterion=f"validation text90 {PRIMARY}, then mean9 {PRIMARY}",
        epoch=best_epoch,raw_ridge=head["ridge"],epochs=EPOCHS,history=history),indent=2))


if __name__=="__main__":
    main()
