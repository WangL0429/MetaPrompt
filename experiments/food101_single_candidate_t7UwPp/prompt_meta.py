"""Second-order cross-condition prompt updates for single-label classification."""
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from food_io import (Food, JOB, condition_masks, encode, extra_indices, grid, load_native, make_schedule,
                     mix_masks, output, save, setup, warm_checkpoint)
from meta_weights import choose_weights
from rule import flat, unit, curvature

EPOCHS, BATCH, SUP, ROUTER = 20, 256, 8, 12
ALPHA = .03


def views(vec, params):
    return [t.view_as(p) for t,p in zip(vec.split([p.numel() for p in params]),params)]


def loss_anchor(z, y, ref):
    divergence = F.kl_div(z.log_softmax(-1), ref.softmax(-1), reduction="batchmean")
    return F.cross_entropy(z,y)+.5*divergence


def inner(net, images, tokens, y, mode):
    total = 0
    for s in range(3):
        idx = np.flatnonzero(mode == s)
        if len(idx):
            z, vi, tx = output(net,images[idx],tokens[idx],s)
            part = F.cross_entropy(z,y[idx])
            if s == 0:
                vi, tx = F.normalize(vi.float()),F.normalize(tx.float())
                center = F.normalize(.2*vi.detach()+.8*tx.detach())
                part = part + F.mse_loss(tx,center) + 2*F.mse_loss(vi,center)
            total = total + len(idx)/len(y)*part
    return total


def step(net, named, batch, refs, rate, rng):
    im,tok,y,indices = batch
    im,tok,y = im.cuda(),tok.cuda(),y.cuda()
    n = len(im)-SUP-ROUTER
    sup = (im[n:n+SUP],tok[n:n+SUP],y[n:n+SUP])
    sm = mix_masks(SUP,rate,rng)
    rm = mix_masks(ROUTER,rate,rng)
    names, params = zip(*named)
    def route(fast):
        z = torch.stack([output(net,im[-ROUTER:],tok[-ROUTER:],s,fast)[0] for s in range(3)],1)
        ix = torch.arange(ROUTER,device=z.device)
        target = indices[-ROUTER:].to(z.device)
        return torch.stack([loss_anchor(z[ix,m],y[-ROUTER:],refs[target,m]) for m in rm])
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
        directions = torch.stack([unit(flat(inner(net,*sup,m),params)) for m in sm])
        with torch.no_grad():
            base = route(None)
            deltas = [route(dict(zip(names,[p-v for p,v in zip(params,views(ALPHA*d,params))])))-base for d in directions]
            weights = directions.new_tensor(choose_weights(torch.stack(deltas).cpu().numpy()))
            shift = views(ALPHA*(weights@directions),params)
        fast = {k:(p-v).detach().requires_grad_() for (k,p),v in zip(named,shift)}
        grads = [torch.zeros_like(directions[0]),torch.zeros_like(directions[0])]
        for side, rows in enumerate(np.array_split(np.arange(n),2)):
            mask = mix_masks(len(rows),rate,rng)[2]
            for state in range(3):
                members = rows[mask==state]
                for j in range(0,len(members),32):
                    sub = members[j:j+32]
                    with torch.autocast("cuda",dtype=torch.float16):
                        z = output(net,im[sub],tok[sub],state,fast if side else None)[0]
                    loss = loss_anchor(z,y[sub],refs[indices[sub],state])
                    v = flat(loss*128,tuple(fast.values()) if side else params)
                    grads[side].add_(v,alpha=len(sub)/(128*n))
        correction = torch.zeros_like(grads[1])
        for w,m in zip(weights.tolist(),sm):
            correction.add_(curvature(inner(net,*sup,m),params,grads[1]), alpha=ALPHA*w)
        grad = grads[0]+grads[1]-correction
        if not grad.isfinite().all():
            raise FloatingPointError("nonfinite second-order gradient")
        return grad, float(correction.norm()/grads[1].norm().clamp_min(1e-12))


def state_logits(d):
    return torch.stack([d[f"base{s}"] for s in range(3)],1)


def main():
    setup()
    init = warm_checkpoint()
    net = load_native(init)
    train, val = Food("train"), Food("val")
    assert set(train.ids).isdisjoint(val.ids)
    teacher = state_logits(encode(net,train,features=False)).cuda()
    masks = condition_masks(val.ids,"val")
    def validation():
        return grid(state_logits(encode(net,val,features=False)),val.labels,masks)[0]
    base = validation()
    learner = net.model.prompt_learner.requires_grad_(True)
    named = list(learner.named_parameters())
    params = [p for _,p in named]
    optimizer = torch.optim.AdamW(params,lr=1e-5,weight_decay=0)
    steps = math.ceil(len(train)/BATCH)
    schedule = make_schedule(steps*EPOCHS,0,"curriculum")
    def snapshot():
        return {k:v.detach().cpu().clone() for k,v in learner.state_dict().items()}
    best, best_epoch, best_state = base,0,snapshot()
    history = []
    resume = JOB/"prompt_resume.pt"
    if resume.exists():
        ck = torch.load(resume,map_location="cpu")
        learner.load_state_dict(ck["current"])
        optimizer.load_state_dict(ck["opt"])
        best, best_epoch, best_state, history = ck["best"],ck["epoch"],ck["best_state"],ck["history"]
    for ep in range(len(history)+1,EPOCHS+1):
        order = np.random.default_rng(1000+ep).permutation(len(train))
        batches = []
        for b,ids in enumerate(np.array_split(order,steps)):
            _,_,seed = schedule[(ep-1)*steps+b]
            extra = extra_indices(len(train),ids,SUP+ROUTER,np.random.default_rng([seed,17]))
            batches.append(np.concatenate([ids,extra]))
        seq = DataLoader(train,batch_sampler=batches,num_workers=6,pin_memory=True,prefetch_factor=1)
        curv = []
        for b,batch in enumerate(seq):
            i=(ep-1)*steps+b
            rate,_,seed = schedule[i]
            grad,r = step(net,named,batch,teacher,rate,np.random.default_rng(seed))
            optimizer.zero_grad(set_to_none=True)
            for p,v in zip(params,views(grad,params)):
                p.grad=v
            optimizer.param_groups[0]["lr"]=1e-5*(.6+.4*math.cos(math.pi*i/(len(schedule)-1)))
            optimizer.step()
            curv.append(r)
        scores=validation()
        if all(scores[k]>=base[k]-.25 for k in scores) and (min(scores.values()),sum(scores.values()))>(min(best.values()),sum(best.values())):
            best,best_epoch,best_state = scores,ep,snapshot()
        history.append(dict(epoch=ep,acc=scores,relative_curvature=float(np.mean(curv))))
        save(dict(current=snapshot(),opt=optimizer.state_dict(),best=best,epoch=best_epoch,best_state=best_state,history=history),resume)
        print(ep,scores["text_90"],best_epoch,flush=True)
    ck=torch.load(init,map_location="cpu")
    bundle={k:ck[k] for k in ("state_dict","hyper_parameters","pytorch-lightning_version")}
    for name,tensor in best_state.items():
        bundle["state_dict"]["model.prompt_learner."+name]=tensor
    bundle["food_prompt_meta"] = dict(selected=best_epoch,epochs=EPOCHS,baseline=base,history=history)
    save(bundle,JOB/"prompt_selected.ckpt")
    (JOB/"prompt_selection.json").write_text(json.dumps(bundle["food_prompt_meta"],indent=2))


if __name__ == "__main__":
    main()
