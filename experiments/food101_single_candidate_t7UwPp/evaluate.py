import json

import numpy as np
import torch
import torch.nn.functional as F

from food_io import CASES, JOB, SEEDS, Food, condition_masks, csv_out, grid, load_native, loader, setup, state_features
from candidate import candidate_scores
from predictor import SemanticPredictor


def load_model(path):
    net=load_native(path)
    ck=torch.load(path,map_location="cpu")
    spec=ck["food_candidate"]
    gen=SemanticPredictor(spec["k"],categorical=spec["categorical"]).cuda().eval().requires_grad_(False)
    gen.load_state_dict(spec["weights"])
    raw=ck["food_image_readout"]
    return net,gen,(raw["weight"].cuda(),raw["bias"].cuda())


@torch.inference_mode()
def infer(model,images,tokens,states,*,compare=False):
    net,gen,(rw,rb)=model
    n=len(images)
    out=torch.empty(n,101,device=images.device)
    without=torch.empty_like(out) if compare else None
    w={1:net.food101_classifier.weight[:,512:],2:net.food101_classifier.weight[:,:512]}
    for state in range(3):
        rows=torch.where(states==state)[0]
        if not len(rows):
            continue
        d=state_features(net,images[rows],tokens[rows],state,raw_only=(state==1 and not compare))
        if state==1:
            out[rows]=F.linear(d["raw"].float(),rw,rb)
        if state==1 and not compare:
            continue
        if state:
            z=candidate_scores(gen,None,d,torch.arange(len(rows),device=rows.device),state,w)
        else:
            z=d["base0"]
        if state!=1:
            out[rows]=z
        if compare:
            without[rows]=z
    return (out,without) if compare else out


def main():
    setup()
    model=load_model(JOB/"food101_selected.ckpt")
    ds=Food("test")
    masks=condition_masks(ds.ids,"test")
    deployed,original=[],[]
    for i,(images,tokens,labels,_) in enumerate(loader(ds)):
        images,tokens=images.cuda(),tokens.cuda()
        batch,base=[],[]
        for state in range(3):
            states=torch.full((len(images),),state,device="cuda",dtype=torch.long)
            a,b=infer(model,images,tokens,states,compare=True)
            batch.append(a.argmax(-1))
            base.append(b.argmax(-1))
        pred=torch.stack(batch,1)
        if i==0:
            states=torch.arange(len(images),device="cuda")%3
            masked_images=images.masked_fill((states==2)[:,None,None,None],0)
            masked_tokens=tokens.masked_fill((states==1)[:,None],0)
            z=infer(model,masked_images,masked_tokens,states)
            assert torch.equal(z.argmax(-1),pred[torch.arange(len(images),device="cuda"),states])
        deployed.append(pred.cpu())
        original.append(torch.stack(base,1).cpu())
    pred,base=torch.cat(deployed),torch.cat(original)
    assert len(pred)==len(ds)==22716 and torch.equal(pred[:,[0,2]],base[:,[0,2]])
    _,a=grid(pred,ds.labels,masks)
    _,b=grid(base,ds.labels,masks)
    rows,details=[],[]
    for mode,rate in CASES:
        values=[]
        for seed in SEEDS:
            v,u=a[mode,rate,seed],b[mode,rate,seed]
            details.append(dict(mode=mode,rate=rate,mask_seed=seed,acc=v,without_readout=u,delta=v-u))
            values.append([v,u,v-u])
        v,u,delta=np.mean(values,axis=0)
        rows.append(dict(mode=mode,rate=rate,acc=v,without_readout=u,delta=delta))
    csv_out(JOB/"test_grid.csv",rows)
    csv_out(JOB/"per_mask.csv",details)
    pure={k:(100*(pred[:,s]==ds.labels).double().mean().item()) for s,k in enumerate(("complete","image_only","text_only"))}
    summary=dict(n_test=len(ds),mask_seeds=SEEDS,mean9=np.mean([r["acc"] for r in rows]),
                 mean9_without_readout=np.mean([r["without_readout"] for r in rows]),pure=pure)
    (JOB/"results.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
