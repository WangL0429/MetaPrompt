"""Movie-paired uncertainty for the prespecified text-90 endpoint and all-scenario point scores."""
import csv
import json
from statistics import mean
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from common import JOB, SOURCE, SEEDS, SCENARIOS, write_csv


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def f1(tp,den):
    return 200*(tp/np.maximum(den,1)).mean(axis=-1)


def sufficient(y,p):
    return np.stack((y*p,y.astype(float)+p),axis=-2)


def by_movie(rows,counts):
    _,compact=np.unique(rows,return_inverse=True)
    matrix=np.zeros((compact.max()+1,*counts.shape[1:]),dtype=float)
    np.add.at(matrix,compact,counts)
    return matrix


def scores_at(counts,multiplicity):
    flat=counts.reshape(len(counts),-1)
    x=(multiplicity@flat).reshape(len(multiplicity),*counts.shape[1:])
    return f1(x[:,:,0],x[:,:,1]).mean(1)


def paired(split,counts):
    n=len(counts["base"])
    rng=np.random.default_rng(1027)
    one=np.ones((1,n))
    points={k:scores_at(v,one).item() for k,v in counts.items()}
    pairs=[("pre_trained","base"),("post_trained","base"),("pre_trained","post_trained"),
           ("pre_selected","base"),("post_selected","base")]
    diffs={p:[] for p in pairs}
    for _ in range(30):
        draws=rng.multinomial(n,np.full(n,1/n),size=50).astype(float)
        measures={k:scores_at(v,draws) for k,v in counts.items()}
        for a,b in pairs:diffs[a,b].extend(measures[a]-measures[b])
    records=[]
    for a,b in pairs:
        lo,hi=np.percentile(diffs[a,b],[2.5,97.5])
        records.append(dict(split=split,metric="text90_macro_F1",candidate=a,reference=b,
                            delta_pp=points[a]-points[b],lower95=lo,upper95=hi,movies=n,draws=1500))
    return records


def diagnostic(split,name,pred,y):
    z=pred[1]
    p=z>0
    c=sufficient(y,p).sum(0)
    return dict(split=split,arm=name,pure_visual_AP=100*average_precision_score(y,z,average="macro"),
                pure_visual_F1=f1(c[0],c[1]),mean_positive_labels=float(p.sum(1).mean()))


def main():
    train_choices={s:json.loads((JOB/f"{s}_selection.json").read_text()) for s in ("pre","post")}
    validation={}
    for stage in ("pre","post"):
        d=torch.load(JOB/f"{stage}_validation.pt",map_location="cpu")
        validation["base"]=d["scores"]["base"].numpy()
        for kind in ("selected","trained"):
            validation[stage+"_"+kind]=d["scores"][kind].numpy()
    y=d["y"].numpy()
    raw_rows=d["rows"].numpy()
    vc={}
    diagnostics=[]
    for k,z in validation.items():
        p=z>0
        c=.1*sufficient(y,p[0])+.9*sufficient(y,p[1])
        vc[k]=by_movie(raw_rows,c[:,None])
        diagnostics.append(diagnostic("validation",k,z,y))
        expected=train_choices["pre"]["baseline"][-1] if k=="base" else train_choices[k.split("_")[0]]["validation"][k.split("_")[1]][-1]
        check=scores_at(vc[k],np.ones((1,len(vc[k])))).item()
        assert abs(check-expected)<2e-5,(k,check,expected)
    intervals=paired("validation",vc)
    data=torch.load(JOB/"test_predictions.pt",map_location="cpu")
    y=data["y"].numpy()
    rows=data["rows"].numpy()
    baseline={(r["missing_type"],round(float(r["missing_rate"])*100),int(r["mask_seed"])):float(r["macro_f1_percent"])
              for r in read_csv(SOURCE.parent/"test.csv")}
    specialist={(r["missing_type"],round(float(r["missing_rate"])*100)):float(r["DPL"])
                for r in read_csv(SOURCE.parent/"test_comparison.csv")}
    results={}
    masks=data["masks"]
    tc={}
    per_seed=[]
    for k in validation:
        z=data["outputs"][data["aliases"].get(k,k)].numpy()
        diagnostics.append(diagnostic("test",k,z,y))
        p=z>0
        text_counts=[]
        for (mode,rate,seed),mask in masks.items():
            pred=p[mask.numpy()[rows],np.arange(len(rows))]
            c=sufficient(y,pred)
            total=c.sum(0)
            value=f1(total[0],total[1])
            if k=="base":
                assert abs(value-baseline[mode,rate,seed])<.0001,(mode,rate,seed,value,baseline[mode,rate,seed])
            per_seed.append(dict(arm=k,mode=mode,rate=rate,seed=seed,macro_F1=value))
            results[k,mode,rate,seed]=value
            if (mode,rate)==("text",90):text_counts.append(c)
        tc[k]=by_movie(rows,np.stack(text_counts,axis=1))
    intervals.extend(paired("test",tc))
    comparison=[]
    for mode,fraction in SCENARIOS:
        rate=round(fraction*100)
        entry=dict(mode=mode,rate=rate,DPL=specialist[mode,rate])
        for k in validation:
            entry[k]=mean(results[k,mode,rate,s] for s in SEEDS)
        comparison.append(entry)
    means={k:mean(row[k] for row in comparison) for k in ("DPL",*validation)}
    comparison.append(dict(mode="average",rate="",**means))
    write_csv("test_grid.csv",comparison)
    write_csv("per_seed.csv",per_seed)
    write_csv("paired_ci.csv",intervals)
    write_csv("visual_readout.csv",diagnostics)
    print(comparison[-1],flush=True)


if __name__=="__main__":
    torch.set_num_threads(2)
    main()
