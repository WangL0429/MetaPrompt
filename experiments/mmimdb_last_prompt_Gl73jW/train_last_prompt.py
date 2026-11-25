"""Joint outer update of the single-candidate module and one shared last-block prompt."""
import json
import math
import time
import numpy as np
import torch
from torch import nn

from fit_semantics import (EPOCHS,STEPS,read,get_weights,masks,loss,support_router,predictions,snapshot)
from last_prompt import LastViews,materialize
from predictor import load_predictor_checkpoint
from semantic_io import JOB,setup,make_schedule,choose_weights,store,json_out,val_grid


@torch.no_grad()
def score_split(net,gen,delta,split,d,w):
    views=LastViews(net,split)
    result=[]
    for ids in np.array_split(np.arange(len(d["y"])),math.ceil(len(d["y"])/128)):
        batch=materialize(d,views,ids,delta)
        result.append(predictions(gen,batch,w)[0])
    return torch.cat(result)


def train():
    setup()
    out=JOB/"joint"
    start=JOB/"previous/text90.ckpt"
    net,gen=load_predictor_checkpoint(start)
    for p in gen.parameters():p.requires_grad_(True)
    gen.mix_score.requires_grad_(False)
    delta=nn.Parameter(torch.zeros(net.model.image_encoder.prompt_length//2,512,device="cuda"))
    named={name:p for name,p in gen.named_parameters() if p.requires_grad}
    params=tuple(named.values())
    opt=torch.optim.AdamW([dict(params=params,lr=1e-4),dict(params=[delta],lr=.001)],weight_decay=.01)
    d,dev=read("train"),read("dev")
    views=LastViews(net,"train")
    w=get_weights()
    rows=d["rows"].cpu().numpy()
    groups=[[] for _ in range(int(rows.max())+1)]
    for i,r in enumerate(rows):groups[r].append(i)
    population=d["y"].float().mean(0)
    pos=((1-population)/population.clamp_min(1e-6)).sqrt().clamp(1,8)
    p=(d["base"]/.5).sigmoid()
    tp=(p*d["y"][:,None]).mean(0)
    pred=p.mean(0)
    schedule=make_schedule(STEPS*EPOCHS,0,"curriculum")
    cal=dev["cal"].cpu()
    labels=dev["y"].cpu()
    previous=json.loads((JOB/"previous/selection.json").read_text())
    control=json.loads((JOB/"control/selection.json").read_text())
    assert previous["picks"]["text90"]==11 and control["recipe"]["initialization"]==str(start.resolve())
    # Full calibration parity at zero is checked before the first update.
    starting=score_split(net,gen,delta,"dev",dev,w)
    old=predictions(gen,dev,w)[0]
    parity=(starting-old).abs().max().item()
    torch.testing.assert_close(starting,old,atol=.012,rtol=.002)
    start_grid=val_grid(starting[cal].transpose(0,1),labels[cal])
    assert np.max(np.abs(np.subtract(start_grid,previous["history"][10]["cal_grid"])))<.025
    history=[]
    if (out/"resume.pt").exists():
        r=torch.load(out/"resume.pt",map_location="cpu")
        gen.load_state_dict(r["weights"])
        delta.data.copy_(r["delta"])
        opt.load_state_dict(r["optimizer"])
        history=r["history"]
    for epoch in range(len(history)+1,EPOCHS+1):
        tic=time.monotonic()
        counts=np.zeros(3,dtype=int)
        prompt_grad=0.
        generator_grad=0.
        perm=np.random.default_rng(1000+epoch).permutation(len(rows))
        for b,ids in enumerate(np.array_split(perm,STEPS)):
            step=(epoch-1)*STEPS+b
            rate,_,seed=schedule[step]
            rng=np.random.default_rng([seed,0])
            support,router=support_router(rows,groups,ids,rng)
            sm=masks(len(support),rate,rng)
            rm=masks(len(router),rate,rng)
            mix=tp.new_tensor([1-rate,rate/2,rate/2])
            den=(mix@pred+population).clamp_min(1e-5)
            stats=((mix@tp)/den,den)
            # Candidate directions remain generator-only. The support features retain their graph
            # so the held-out loss also differentiates through the inner update into the prompt.
            sd=materialize(d,views,support,delta)
            si=np.arange(len(support))
            directions=[]
            for m in sm:
                l=loss(gen,None,sd,si,m,w,stats,pos,True,False,semantic_weight=0.)
                grads=torch.autograd.grad(l,params,create_graph=True,allow_unused=True)
                gs=[torch.zeros_like(p) if g is None else g for p,g in zip(params,grads)]
                norm=torch.cat([g.flatten() for g in gs]).norm().clamp_min(1e-8)
                directions.append([g/norm for g in gs])
            with torch.no_grad():
                rd=materialize(d,views,router,delta)
                ri=np.arange(len(router))
                def route(parameters):
                    return torch.stack([loss(gen,parameters,rd,ri,m,w,stats,pos,False,True) for m in rm])
                ref=route(None)
                moves=[route({n:p-.03*g for (n,p),g in zip(named.items(),dr)})-ref for dr in directions]
                weights=choose_weights(torch.stack(moves).cpu().numpy()).tolist()
            fast={n:p-.03*sum(wi*dr[i] for wi,dr in zip(weights,directions))
                  for i,(n,p) in enumerate(named.items())}
            value=0.
            for side,part in enumerate(np.array_split(ids,2)):
                mode=masks(len(part),rate,rng)[2]
                states=mode.cpu().numpy()
                counts+=np.bincount(states,minlength=3)
                qd=materialize(d,views,part,delta,mode=states)
                value+=len(part)/len(ids)*loss(gen,None if side==0 else fast,qd,np.arange(len(part)),
                                                mode,w,stats,pos,True,True,semantic_weight=0.)
            opt.zero_grad(set_to_none=True)
            value.backward()
            prompt_grad+=delta.grad.norm().item()
            generator_grad+=nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True).item()
            nn.utils.clip_grad_norm_([delta],2.,error_if_nonfinite=True)
            scale=.2+.4*(1+math.cos(math.pi*step/(STEPS*EPOCHS-1)))
            opt.param_groups[0]["lr"]=1e-4*scale
            opt.param_groups[1]["lr"]=.001*scale
            opt.step()
        assert counts.tolist()==control["history"][epoch-1]["query_state_counts"]
        assert prompt_grad>0 and generator_grad>0 and delta.isfinite().all()
        z=score_split(net,gen,delta,"dev",dev,w)
        grid=val_grid(z[cal].transpose(0,1),labels[cal])
        assert np.isfinite(grid).all()
        hist=dict(epoch=epoch,cal_grid=grid,seconds=time.monotonic()-tic,query_state_counts=counts.tolist(),
                  prompt_gradient_norm=prompt_grad/STEPS,generator_gradient_norm=generator_grad/STEPS,
                  prompt_norm=delta.norm().item())
        history.append(hist)
        spec=dict(weights=snapshot(gen),delta=delta.detach().cpu())
        store(out/f"epoch_{epoch:02}.pt",spec)
        store(out/"resume.pt",dict(**spec,optimizer=opt.state_dict(),history=history))
        if epoch==1:json_out(out/"first_epoch.json",hist)
        print(epoch,round(float(np.mean(grid)),4),round(grid[-1],4),round(hist["seconds"],1),flush=True)
    picks={"mean9":max(history,key=lambda r:(np.mean(r["cal_grid"]),r["cal_grid"][-1]))["epoch"],
           "text90":max(history,key=lambda r:(r["cal_grid"][-1],np.mean(r["cal_grid"])))["epoch"]}
    for epoch in set(picks.values()):
        ck=torch.load(start,map_location="cpu")
        state=torch.load(out/f"epoch_{epoch:02}.pt",map_location="cpu")
        ck["semantic_predictor"].update(weights=state["weights"],last_prompt=state["delta"],epoch=epoch,
            recipe=dict(initialization=str(start.resolve()),prompt_lr=.001,generator_lr=.0001))
        store(out/f"deploy_epoch{epoch}.ckpt",ck)
        del ck
    for name,epoch in picks.items():
        link=out/(name+".ckpt")
        if not link.exists():link.symlink_to(f"deploy_epoch{epoch}.ckpt")
    json_out(out/"selection.json",dict(picks=picks,epochs=EPOCHS,steps=STEPS,history=history,
            init_dev_max_error=parity,init_grid=start_grid,generator_trainable=sum(p.numel() for p in params),
            trainable=sum(p.numel() for p in params)+delta.numel(),last_prompt_params=delta.numel()))


if __name__=="__main__":
    train()
