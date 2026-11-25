import json

import numpy as np
import torch
import torch.nn.functional as F

from single_label import BINARY, N_CLASSES, PRIMARY, TAG, TASK, decision, head, measure
from food_io import CASES, JOB, SEEDS, SingleLabelDataset, condition_masks, csv_out, grid, load_native, loader, setup, state_features
from candidate import candidate_scores
from predictor import SemanticPredictor


def load_model(path):
    net=load_native(path)
    ck=torch.load(path,map_location="cpu")
    spec=ck[f"{TAG}_candidate"]
    gen=SemanticPredictor(spec["k"],categorical=spec["categorical"]).cuda().eval().requires_grad_(False)
    gen.load_state_dict(spec["weights"])
    raw=ck[f"{TAG}_image_readout"]
    return net,gen,(raw["weight"].cuda(),raw["bias"].cuda())


@torch.inference_mode()
def infer(model,images,tokens,states,*,compare=False):
    net,gen,(rw,rb)=model
    n=len(images)
    out=torch.empty(n,N_CLASSES,device=images.device)
    without=torch.empty_like(out) if compare else None
    w={1:head(net).weight[:,512:],2:head(net).weight[:,:512]}
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
    model=load_model(JOB/f"{TASK}_selected.ckpt")
    ds=SingleLabelDataset("test")
    masks=condition_masks(ds.ids,"test")
    deployed,original=[],[]
    for i,(images,tokens,labels,_) in enumerate(loader(ds)):
        images,tokens=images.cuda(),tokens.cuda()
        batch,base=[],[]
        for state in range(3):
            states=torch.full((len(images),),state,device="cuda",dtype=torch.long)
            a,b=infer(model,images,tokens,states,compare=True)
            batch.append(decision(a))
            base.append(decision(b))
        pred=torch.stack(batch,1)
        if i==0:
            states=torch.arange(len(images),device="cuda")%3
            masked_images=images.masked_fill((states==2)[:,None,None,None],0)
            masked_tokens=tokens.masked_fill((states==1)[:,None],0)
            z=infer(model,masked_images,masked_tokens,states)
            expected=pred[torch.arange(len(images),device="cuda"),states]
            if BINARY:
                torch.testing.assert_close(decision(z),expected,rtol=.01,atol=.025)
            else:
                assert torch.equal(decision(z),expected)
        deployed.append(pred.cpu())
        original.append(torch.stack(base,1).cpu())
    pred,base=torch.cat(deployed),torch.cat(original)
    assert len(pred)==len(ds)==(1000 if BINARY else 22716) and torch.equal(pred[:,[0,2]],base[:,[0,2]])
    pure_names=("complete","image_only","text_only")
    metrics=(PRIMARY,"acc") if BINARY else (PRIMARY,)
    summary=dict(n_test=len(ds),mask_seeds=SEEDS,primary=PRIMARY)
    for metric in metrics:
        _,a=grid(pred,ds.labels,masks,metric)
        _,b=grid(base,ds.labels,masks,metric)
        rows,details=[],[]
        for mode,rate in CASES:
            values=[]
            for seed in SEEDS:
                v,u=a[mode,rate,seed],b[mode,rate,seed]
                details.append({"mode":mode,"rate":rate,"mask_seed":seed,metric:v,"without_readout":u,"delta":v-u})
                values.append([v,u,v-u])
            v,u,delta=np.mean(values,axis=0)
            rows.append({"mode":mode,"rate":rate,metric:v,"without_readout":u,"delta":delta})
        suffix="" if metric==PRIMARY else "_"+metric
        csv_out(JOB/f"test_grid{suffix}.csv",rows)
        csv_out(JOB/f"per_mask{suffix}.csv",details)
        summary.update({"mean9"+suffix:np.mean([r[metric] for r in rows]),
                        "mean9_without_readout"+suffix:np.mean([r["without_readout"] for r in rows]),
                        "pure"+suffix:{k:measure(pred[:,s],ds.labels,metric) for s,k in enumerate(pure_names)},
                        "pure_without_readout"+suffix:{k:measure(base[:,s],ds.labels,metric) for s,k in enumerate(pure_names)}})
    (JOB/"results.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
