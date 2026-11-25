"""Fit one correction to the existing binary classifier, using only train and validation."""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from candidate import candidate_scores, predict
from food_io import JOB, SingleLabelDataset, condition_masks, csv_out, grid, loader, output, save, setup
from evaluate import load_model
from single_label import head

OUT = Path(__file__).resolve().parent
SOURCE = JOB / "hatememes_selected.ckpt"
KEY = "hatememes_classifier."


@torch.no_grad()
def features(model,split):
    net,gen,_ = model
    ds=SingleLabelDataset(split)
    d={k:v.cuda() for k,v in torch.load(JOB/f"{split}_features.pt",map_location="cpu").items()}
    y=d["y"]
    assert torch.equal(y.cpu(),ds.labels) and gen.k==1
    chunks=[]
    for img,tok,_,_ in loader(ds):
        with torch.autocast("cuda",dtype=torch.float16):
            _,im,tx=output(net,img.cuda(),tok.cuda(),0)
            chunks.append(torch.cat((im,tx),1).half())
    complete=torch.cat(chunks)
    n=len(ds)
    observed=torch.zeros(n,3,1024,device="cuda",dtype=torch.float16)
    observed[:,0]=complete
    observed[:,1,:512]=d["h1"]
    observed[:,2,512:]=d["h2"]
    vectors=torch.empty(n,2,512,device="cuda")
    gates=torch.empty(n,2,device="cuda")
    w={1:head(net).weight[:,512:],2:head(net).weight[:,:512]}
    for s in (1,2):
        for ix in torch.arange(n,device="cuda").split(256):
            args=(d[f"h{s}"][ix],d[f"x{s}"][ix],d[f"mask{s}"][ix],s,d[f"base{s}"][ix],w[s])
            _,(direction,magnitude,_,gate)=gen(*args)
            vectors[ix,s-1]=direction[:,0]*magnitude[:,0,None]
            gates[ix,s-1]=gate[:,0]
    data=(observed,vectors,gates)
    base=scores(data,head(net).weight,head(net).bias,True)
    reference=predict(gen,d,w).to(base.device)
    torch.testing.assert_close(base.double(),reference,atol=.008,rtol=.002)
    return data,y,ds.ids


def scores(data,w,b,deploy=False):
    obs,v,gates=data
    if deploy:
        z=F.linear(obs,w.half(),b.half()).float()
    else:
        z=F.linear(obs.float(),w,b)
    # The frozen generator emits one candidate and a mixing probability.
    add=torch.stack((F.linear(v[:,0],w[:,512:]),F.linear(v[:,1],w[:,:512])),1)
    both=torch.logaddexp(F.logsigmoid(-gates)[:,:,None]+z[:,1:].log_softmax(-1),
                         F.logsigmoid(gates)[:,:,None]+(z[:,1:]+add).log_softmax(-1))
    logits=torch.cat((z[:,:1],both),1)
    return logits[:,:,1]-logits[:,:,0]


def pairs(y):
    rng=np.random.default_rng(803)
    pos=np.flatnonzero(y.cpu().numpy()==1)
    neg=np.flatnonzero(y.cpu().numpy()==0)
    # Equal representation for each of the nine positive-state/negative-state combinations.
    p,n=[],[]
    for s in range(3):
        for t in range(3):
            p.append(3*rng.choice(pos,8192)+s)
            n.append(3*rng.choice(neg,8192)+t)
    return tuple(torch.as_tensor(np.concatenate(x),device=y.device) for x in (p,n))


def fit(data,y,w,b):
    obs=data[0]
    scale=torch.cat((obs[:,[0,1],:512].float().square().mean((0,1)),
                     obs[:,[0,2],512:].float().square().mean((0,1)))).sqrt().clamp_min(.01)
    beta=torch.zeros_like(scale,requires_grad=True)
    offset=torch.zeros((),device=y.device,requires_grad=True)
    signs=w.new_tensor([-.5,.5])
    ip,ineg=pairs(y)
    opt=torch.optim.LBFGS([beta,offset],max_iter=150,lr=1.,tolerance_grad=1e-6,line_search_fn="strong_wolfe")
    target=y[:,None].float().expand(-1,3)
    def closure():
        opt.zero_grad()
        z=scores(data,w+signs[:,None]*(beta/scale),b+signs*offset)
        flat=z.flatten()
        ranking=F.softplus(flat[ineg]-flat[ip]).mean()
        ce=F.binary_cross_entropy_with_logits(z,target,reduction="none")
        balanced=.5*(ce[y==1].mean()+ce[y==0].mean())
        loss=ranking+.5*balanced+.15*beta.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    final=float(closure().detach())
    assert math.isfinite(final)
    return signs[:,None]*(beta.detach()/scale),signs*offset.detach(),final


def main():
    setup()
    model=load_model(SOURCE)
    net=model[0]
    tr,y,ids=features(model,"train")
    val,yv,valids=features(model,"val")
    assert set(ids).isdisjoint(valids)
    masks=condition_masks(valids,"val")
    w,b=head(net).weight.detach(),head(net).bias.detach()
    dw,db,obj=fit(tr,y,w,b)
    rows=[]
    selected=None
    for step in (0.,.125,.25,.5,1.):
        z=scores(val,w+step*dw,b+step*db,True)
        g,_=grid(z.double(),yv,masks)
        mean=float(np.mean(list(g.values())))
        r=dict(step=step,mean9=mean,**g)
        rows.append(r)
        # Retain the original text90 tie-breaking order; protect the weak image90 group and mean.
        if step==0 or (mean>=rows[0]["mean9"] and g["image_90"]>=rows[0]["image_90"] and
                       (g["text_90"],mean)>(selected["text_90"],selected["mean9"])):
            selected=r
    csv_out(OUT/"validation.csv",rows)
    alpha=selected["step"]
    ck=torch.load(SOURCE,map_location="cpu")
    ck.pop("hatememes_image_readout")
    ck["state_dict"][KEY+"weight"]=(w+alpha*dw).cpu()
    ck["state_dict"][KEY+"bias"]=(b+alpha*db).cpu()
    summary=dict(source=str(SOURCE),train=len(ids),validation=len(valids),selected=selected,baseline=rows[0],
                 trained_margin_coefficients=dw.shape[1]+1,added_parameters=0,train_loss=obj)
    ck["rank_head"]=summary
    save(ck,OUT/"selected.ckpt")
    (OUT/"selection.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
