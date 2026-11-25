"""Fit a rank-two residual for both projections with a common output basis, then merge."""
import math
import json
import runpy
import sys
import numpy as np
import torch
import torch.nn.functional as F

from extract import JOB, BASE, SHARED
sys.path.insert(0,str(BASE))
from model_io import choose_weights
from common import val_grid
from clip.datamodules.missing_schedule import make_schedule

risk = runpy.run_path(str(BASE/"train.py"))["risk"]
VIS = "model.image_encoder.proj"
TXT = "model.text_encoder.text_projection"


def read(split):
    return {k:v.cuda() for k,v in torch.load(JOB/("lowrank_"+split+".pt"),map_location="cpu").items()}


class Readout(torch.nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.register_buffer("pv",sd[VIS].cuda().float())
        self.register_buffer("pt",sd[TXT].cuda().float())
        self.register_buffer("weight",sd["mmimdb_classifier.weight"].cuda().float())
        self.register_buffer("bias",sd["mmimdb_classifier.bias"].cuda().float())
        self.u_v=torch.nn.Parameter(torch.zeros(768,2,device="cuda"))
        self.u_t=torch.nn.Parameter(torch.zeros(512,2,device="cuda"))
        self.basis=torch.nn.Parameter(torch.randn(2,512,device="cuda")/512**.5)

    def matrices(self):
        return self.pv+self.u_v@self.basis,self.pt+self.u_t@self.basis

    def forward(self,d,index,state):
        film=d["rows"][index]
        v=d["v0"][film].clone()
        keep1=state==1
        v[keep1]=d["v1"][film[keep1]]
        v[state==2]=0
        t=d["t0"][index].clone()
        keep2=state==2
        t[keep2]=d["t2"][index[keep2]]
        t[keep1]=0
        pv,pt=self.matrices()
        fv=v@pv.half()
        ft=t@pt.half()
        return F.linear(torch.cat((fv,ft),1),self.weight.half(),self.bias.half()).float()


@torch.no_grad()
def metrics(net,data,calibration=True):
    keep=data["cal"] if calibration else ~data["cal"]
    ix=torch.nonzero(keep).flatten()
    scores=[]
    for state in (0,1,2):
        z=[net(data,i,torch.full_like(i,state)) for i in ix.split(64)]
        scores.append(torch.cat(z).cpu())
    return val_grid(torch.stack(scores),data["y"][ix].cpu())


def make_view(data,items,mode,rate,rng):
    ix=torch.as_tensor(items,device="cuda")
    n=len(ix)
    missing=round(n*rate)
    order=rng.permutation(n)[:missing]
    if mode=="both":
        half=(missing+int(rng.integers(2)))//2
    else:
        half=missing if mode=="text" else 0
    st=np.zeros(n,dtype=np.int64)
    st[order[:half]]=1
    st[order[half:]]=2
    st=torch.from_numpy(st).cuda()
    row=data["rows"][ix]
    v=torch.where((st==0)[:,None],data["v0"][row],data["v1"][row]).float()
    t=torch.where((st==0)[:,None],data["t0"][ix],data["t2"][ix]).float()
    return (v.masked_fill_((st==2)[:,None],0),t.masked_fill_((st==1)[:,None],0),
            data["ref"][st,ix],data["y"][ix].float())


def update_logits(view,params,net):
    v,t,ref,_=view
    uv,ut,basis=params
    wv,wt=net.weight.split(512,dim=1)
    return ref+(v@uv)@(basis@wv.T)+(t@ut)@(basis@wt.T)


def main():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(SHARED,map_location="cpu")
    sd=ck["state_dict"]
    net=Readout(sd)
    assert sum(p.numel() for p in net.parameters())==3584
    train,dev=read("train"),read("dev")
    n=len(train["y"])
    base=metrics(net,dev)
    valid_base=metrics(net,dev,False)
    ref=json.loads((SHARED.parent/"selection.json").read_text())["selected"]
    assert abs(min(valid_base)-ref["min_f1"])<.08, (valid_base,ref)
    schedule=make_schedule(20*64,0,"curriculum")
    frequency=train["y"].float().mean(0)
    pos_weight=((1-frequency)/frequency.clamp_min(1e-6)).sqrt().clamp(1,8)
    ids=torch.arange(n,device="cuda")
    with torch.no_grad():
        train["ref"]=torch.stack([torch.cat([net(train,ix,torch.full_like(ix,state))
                                             for ix in ids.split(64)]) for state in (0,1,2)])
        prob=(train["ref"]/.5).sigmoid()
        tp=(prob*train["y"].unsqueeze(0)).mean(1)
        positive=prob.mean(1)
    films=train["rows"].cpu().numpy()
    by_film=[[] for _ in range(int(films.max())+1)]
    for i,film in enumerate(films): by_film[film].append(i)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=0)
    history=[]
    best_key=(min(base),sum(base)/9)
    selected_epoch=0
    best=None
    nonzero_key=None
    nonzero_epoch=None
    nonzero=None
    history.append(dict(epoch=0,eligible=True,grid=base))
    params=tuple(net.parameters())
    # Replay only the calibration-chosen trained candidate for a separate diagnostic test.
    replay = "--export-trained" in sys.argv
    first_run = json.loads((JOB/"lowrank_selection.json").read_text()) if replay else None
    epochs = first_run["best_nonzero_epoch"] if replay else 20
    for epoch in range(epochs):
        batches=np.array_split(np.random.default_rng(1001+epoch).permutation(n),64)
        for step,batch in enumerate(batches):
            tick=epoch*64+step
            rate,_,seed=schedule[tick]
            rng=np.random.default_rng([seed,0])
            mix=tp.new_tensor([1-rate,rate/2,rate/2])
            denom=(mix@positive+frequency).clamp_min(1e-5)
            stats=((mix@tp)/denom,denom)
            excluded=set(films[batch].tolist())
            extra=[]
            while len(extra)<20:
                f=int(rng.integers(len(by_film)))
                if f not in excluded:
                    excluded.add(f)
                    extra.append(int(rng.choice(by_film[f])))
            support,router=extra[:8],extra[8:]
            directions=[]
            for mode in ("text","image","both"):
                view=make_view(train,support,mode,rate,rng)
                g=torch.autograd.grad(F.binary_cross_entropy_with_logits(update_logits(view,params,net),view[3]),
                                      params,create_graph=True)
                norm=torch.cat([p.reshape(-1) for p in g]).norm().clamp_min(1e-9)
                directions.append(tuple(p/norm for p in g))
            router_views=[make_view(train,router,mode,rate,rng) for mode in ("text","image","both")]
            with torch.no_grad():
                def route(p):
                    return torch.stack([risk(update_logits(v,p,net),v[3],v[2],pos_weight,stats) for v in router_views])
                origin=route(params)
                differences=torch.stack([route(tuple(p-.03*d for p,d in zip(params,dr)))-origin for dr in directions])
                weights=choose_weights(differences.cpu().numpy()).tolist()
            fast=tuple(p-.03*sum(w*dr[i] for w,dr in zip(weights,directions)) for i,p in enumerate(params))
            query_loss=0.
            for part,values in zip(np.array_split(batch,2),(params,fast)):
                v=make_view(train,part,"both",rate,rng)
                query_loss+=risk(update_logits(v,values,net),v[3],v[2],pos_weight,stats)*(len(part)/len(batch))
            opt.zero_grad(set_to_none=True)
            query_loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.param_groups[0]["lr"]=3e-4*(.2+.4*(1+math.cos(math.pi*tick/(len(schedule)-1))))
            opt.step()
        grid=metrics(net,dev)
        if replay and epoch+1==epochs:
            np.testing.assert_allclose(grid,first_run["history"][epochs]["grid"],rtol=0,atol=1e-5)
            np.testing.assert_allclose(metrics(net,dev,False),first_run["valid_nonzero"],rtol=0,atol=1e-5)
            with torch.no_grad():
                for key,matrix in zip((VIS,TXT),net.matrices()):
                    sd[key]=matrix.to(device="cpu",dtype=sd[key].dtype)
            ck["paired_projection_fit"]=dict(source=str(SHARED),rank=2,trainable=3584,epoch=epochs)
            torch.save(ck, JOB/"lowrank_trained.ckpt")
            return
        eligible=min(new-old for new,old in zip(grid,base))>=-.25
        history.append(dict(epoch=epoch+1,eligible=eligible,grid=grid))
        if epoch==0:
            print("First full epoch validated",min(grid),sum(grid)/9,flush=True)
        key=(min(grid),sum(grid)/9)
        diagnostic_key=(eligible,*key)
        if nonzero_key is None or diagnostic_key>nonzero_key:
            nonzero_key,nonzero_epoch=diagnostic_key,epoch+1
            nonzero={k:p.detach().clone() for k,p in net.named_parameters()}
        if eligible and key>best_key:
            best_key,selected_epoch=key,epoch+1
            best={k:p.detach().clone() for k,p in net.named_parameters()}
    with torch.no_grad():
        for k,p in net.named_parameters(): p.copy_(nonzero[k])
        valid_nonzero=metrics(net,dev,False)
        for k,p in net.named_parameters():
            if best is not None:p.copy_(best[k])
        if best is None:
            net.u_v.zero_()
            net.u_t.zero_()
        heldout=metrics(net,dev,False)
        chosen=metrics(net,dev)
        pv,pt=net.matrices()
        assert all(abs(a-b)<.0001 for a,b in zip(chosen,history[selected_epoch]["grid"]))
        for key,matrix in ((VIS,pv),(TXT,pt)):
            sd[key]=matrix.to(device="cpu",dtype=sd[key].dtype)
    ck["paired_projection_fit"]=dict(source=str(SHARED),rank=2,trainable=3584,epoch=selected_epoch)
    if selected_epoch:
        torch.save(ck,JOB/"lowrank_selected.ckpt")
    else:
        (JOB/"lowrank_selected.ckpt").symlink_to(SHARED)
    record=dict(selected=selected_epoch,history=history,cal_base=base,cal_selected=chosen,
                valid_base=valid_base,valid_selected=heldout,params=3584,
                trainable_names=[k for k,_ in net.named_parameters()],best_nonzero_epoch=nonzero_epoch,
                valid_nonzero=valid_nonzero,steps=1280,inner_step=.03,seed=0,outer_lr=3e-4)
    (JOB/"lowrank_selection.json").write_text(json.dumps(record,indent=2))
    print("rank-two training and validation complete",selected_epoch,heldout,flush=True)


if __name__=="__main__":
    main()

