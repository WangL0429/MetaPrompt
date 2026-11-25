"""Test only the locked validation choice against the two archived ablations."""
import argparse
import json

import numpy as np
import torch

from evaluate import infer,load_model
from food_io import CASES,JOB,SEEDS,TRAIN_SEED,SingleLabelDataset,condition_masks,csv_out,grid,loader,setup
from rank_head import OUT,SOURCE
from single_label import decision,measure


def paired_intervals(scores,y,masks,draws=1000):
    # Resample meme IDs jointly across models and mask seeds. Count concordant pairs with half credit for ties.
    yn=y.numpy()
    n=len(yn)
    mult=np.random.default_rng(20261123).multinomial(n,np.full(n,1/n),size=draws)
    idx=np.arange(n)
    totals={}
    for model in ("candidate","ranked"):
        effect={}
        for key,mask in masks.items():
            z=scores[model].numpy()[idx,mask.numpy()]
            order=np.argsort(z,kind="stable")
            labels=yn[order]
            groups=np.r_[0,1+np.flatnonzero(np.diff(z[order]))]
            pieces=[]
            for batch in np.array_split(mult,20):
                weights=batch[:,order]
                p=np.add.reduceat(weights*labels,groups,axis=1)
                q=np.add.reduceat(weights*(1-labels),groups,axis=1)
                below=q.cumsum(axis=1)-q*.5
                pieces.append(100*(p*below).sum(1)/(p.sum(1)*q.sum(1)))
            effect[key]=np.concatenate(pieces)
        totals[model]=effect
    diff={k:totals["ranked"][k]-totals["candidate"][k] for k in masks}
    groups={"mean9":list(masks),"text_90":[("text",90,s) for s in SEEDS],"image_90":[("image",90,s) for s in SEEDS]}
    return {g:np.percentile(np.mean([diff[k] for k in keys],axis=0),[2.5,97.5]).tolist() for g,keys in groups.items()}


@torch.inference_mode()
def main(replication=False):
    setup()
    selection=json.loads((OUT/"selection.json").read_text())
    if selection["selected"]["step"]==0 and not replication:
        print("Original candidate retained on validation; no new test evaluation.")
        return
    old,new=load_model(SOURCE),load_model(OUT/"selected.ckpt")
    assert old[2] is not None and new[2] is None
    ds=SingleLabelDataset("test")
    assert len(ds)==1000
    masks=condition_masks(ds.ids,"test")
    states={k:[] for k in ("raw","candidate","ranked")}
    for i,(im,tk,_,_) in enumerate(loader(ds)):
        im,tk=im.cuda(),tk.cuda()
        pieces={k:[] for k in states}
        for s in range(3):
            mode=torch.full((len(tk),),s,device="cuda")
            a,b=infer(old,im,tk,mode,compare=True)
            c=infer(new,im,tk,mode)
            for k,z in zip(pieces,(a,b,c)):
                pieces[k].append(decision(z))
        for k,v in pieces.items():
            states[k].append(torch.stack(v,1).cpu())
        if i==0:
            mode=torch.arange(len(tk),device="cuda")%3
            mixed=decision(infer(new,im.masked_fill((mode==2)[:,None,None,None],0),tk.masked_fill((mode==1)[:,None],0),mode))
            expected=states["ranked"][0][torch.arange(len(tk)),mode.cpu()]
            torch.testing.assert_close(mixed.cpu(),expected,atol=.025,rtol=.01)
    z={k:torch.cat(v) for k,v in states.items()}
    if selection["selected"]["step"]==0:
        torch.testing.assert_close(z["ranked"],z["candidate"],atol=0,rtol=0)
    rows,per=[],[]
    summary={"pure":{},"mean9":{},"n":len(ds),"seed":TRAIN_SEED}
    for metric in ("auroc","acc"):
        all_scores={k:grid(v,ds.labels,masks,metric) for k,v in z.items()}
        out={k:float(np.mean(list(g[0].values()))) for k,g in all_scores.items()}
        summary["mean9"][metric]=out
        if not replication:
            archived=json.loads((JOB/"results.json").read_text())
            suf="_acc" if metric=="acc" else ""
            for k,field in (("raw","mean9"),("candidate","mean9_without_readout")):
                assert abs(out[k]-archived[field+suf])<.001,(k,out,archived)
        summary["pure"][metric]={k:[measure(v[:,s],ds.labels,metric) for s in range(3)] for k,v in z.items()}
        for m,r in CASES:
            vals={k:g[0][f"{m}_{r}"] for k,g in all_scores.items()}
            rows.append(dict(metric=metric,mode=m,rate=r,**vals,delta=vals["ranked"]-vals["candidate"]))
            for seed in SEEDS:
                vals={k:g[1][m,r,seed] for k,g in all_scores.items()}
                per.append(dict(metric=metric,mode=m,rate=r,mask=seed,**vals,delta=vals["ranked"]-vals["candidate"]))
    summary["paired95_auroc_vs_candidate"]=paired_intervals(z,ds.labels,masks)
    csv_out(OUT/"test_grid.csv",rows)
    csv_out(OUT/"per_mask.csv",per)
    (OUT/"results.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--replication",action="store_true",help="Test both arms together, including zero-step selections")
    main(parser.parse_args().replication)
