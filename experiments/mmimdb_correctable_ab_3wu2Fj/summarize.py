"""Audit matched training and report ranking, the nine deployment scenarios and paired intervals."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score
from assess import JOB,SOURCE,CACHED,baseline
from common import val_grid
from evaluate_mmimdb import MODES,RATES,SEEDS


def table(name,rows):
    path=JOB/name
    with path.with_suffix(".part").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    path.with_suffix(".part").replace(path)


def check_history():
    a,b=[json.loads((JOB/arm/"selection.json").read_text()) for arm in ("A","B")]
    assert a["source"]==b["source"] and a["seed"]==b["seed"] and a["trainable"]==b["trainable"]
    assert len(a["history"])==len(b["history"])==21
    for x,y in zip(a["history"][1:],b["history"][1:]):
        assert x["slots_sha256"]==y["slots_sha256"] and x["paired_stats"]==y["paired_stats"]
    return a,b


def aggregate(rows,array):
    _,index=np.unique(rows,return_inverse=True)
    sums=np.zeros((max(index)+1,*array.shape[1:]),dtype=np.float64)
    np.add.at(sums,index,array)
    return sums


def ranking(scores,y,rows):
    n=aggregate(rows,np.ones((len(rows),1)))
    yy=aggregate(rows,y)/n
    assert np.all((yy==0)|(yy==1))
    records=[]
    for state,nm in ((1,"image_only"),(2,"text_only")):
        movies={name:aggregate(rows,z[state])/n for name,z in scores.items()}
        for name,z in movies.items():
            gains=[]
            for c in range(y.shape[1]):
                pos=yy[:,c].astype(bool)
                def hits(s):
                    margin=s[pos,c,None]-s[~pos,c][None,:]
                    return (margin>0)+.5*(margin==0)
                d=hits(z)-hits(movies["base"])
                gains.append((d.clip(min=0).mean(),(-d).clip(min=0).mean()))
            fix,breakage=100*np.mean(gains,axis=0)
            records.append(dict(arm=name,state=nm,movies=len(n),macro_AP=100*average_precision_score(yy,z),
                                macro_AUC=100*roc_auc_score(yy,z),base_wrong_fixed_pp=fix,
                                base_correct_lost_pp=breakage))
    return records


def ranking_intervals(scores,y,rows):
    """Bootstrap independent movies with tied class scores receiving half credit."""
    weight=aggregate(rows,np.ones((len(rows),1)))
    labels=(aggregate(rows,y)/weight).astype(bool)
    n=len(labels)
    draws=np.random.default_rng(902).multinomial(n,np.full(n,1/n),size=1500).astype(float)
    out=[]
    for state in (1,2):
        samples={}
        point={}
        for arm,z in scores.items():
            movie_z=aggregate(rows,z[state])/weight
            point[arm]=100*roc_auc_score(labels,movie_z)
            total=np.zeros(len(draws))
            defined=np.zeros(len(draws))
            for j in range(labels.shape[1]):
                order=np.argsort(movie_z[:,j],kind="stable")
                starts=np.r_[0,np.flatnonzero(np.diff(movie_z[order,j]))+1]
                target=labels[order,j]
                for lo in range(0,len(draws),50):
                    sub=slice(lo,lo+50)
                    w=draws[sub][:,order]
                    pos=np.add.reduceat(w*target,starts,axis=1)
                    neg=np.add.reduceat(w*~target,starts,axis=1)
                    denom=pos.sum(1)*neg.sum(1)
                    credit=(pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1)
                    total[sub]+=credit/np.maximum(denom,1)
                    defined[sub]+=denom>0
            samples[arm]=100*total/defined
        for a,b in (("B_selected","A_selected"),("B_selected","base"),("B_trained","A_trained")):
            lo,hi=np.percentile(samples[a]-samples[b],[2.5,97.5])
            out.append(dict(state=("image_only" if state==1 else "text_only"),metric="macro_AUC",
                            arm=a,ref=b,delta_pp=point[a]-point[b],lower95=lo,upper95=hi,
                            movies=n,repeats=len(draws)))
    return out


def counts(pred,y):
    return np.stack((pred*y,pred.astype(float)+y),axis=-2)


def paired(split,cc):
    n=len(cc["base"])
    def f(array,draws):
        x=(draws@array.reshape(n,-1)).reshape(len(draws),*array.shape[1:])
        return 200*(x[:,:,0]/np.maximum(x[:,:,1],1)).mean(axis=(1,2))
    comparisons=[("B_selected","A_selected"),("A_selected","base"),("B_selected","base"),
                 ("B_trained","A_trained"),("B_trained","base"),("A_trained","base")]
    point={k:f(arr,np.ones((1,n)))[0] for k,arr in cc.items()}
    rng=np.random.default_rng(117)
    series={p:[] for p in comparisons}
    for _ in range(30):
        draws=rng.multinomial(n,np.full(n,1/n),size=50).astype(float)
        vals={k:f(arr,draws) for k,arr in cc.items()}
        for a,b in comparisons:series[a,b].extend(vals[a]-vals[b])
    results=[]
    for a,b in comparisons:
        lo,hi=np.percentile(series[a,b],[2.5,97.5])
        results.append(dict(split=split,endpoint="text90_macro_F1",arm=a,ref=b,
                            delta_pp=point[a]-point[b],lo95=lo,hi95=hi,movies=n,repeats=1500))
    return results


def report(check_train=True):
    selection=check_history() if check_train else None
    outputs={arm:torch.load(JOB/arm/"predictions.pt",map_location="cpu") for arm in ("A","B")}
    summary,intervals,ranking_rows,per_seed=[],[],[],[]
    with (SOURCE.parent/"test_comparison.csv").open() as f:
        refs={(r["missing_type"],int(round(float(r["missing_rate"])*100))):r for r in csv.DictReader(f)}
    for split in ("dev","test"):
        native,rows,y=baseline(split)
        rows=rows.numpy()
        labels=y.numpy()
        scores={"base":native.numpy()}
        for arm,data in outputs.items():
            np.testing.assert_array_equal(data[split]["base"].numpy(),scores["base"])
            for name in ("selected","trained"):scores[arm+"_"+name]=data[split][name].numpy()
        sufficient={}
        if split=="dev":
            ranking_rows=ranking(scores,labels,rows)
            table("unseen_ranking_ci.csv",ranking_intervals(scores,labels,rows))
            for name,z in scores.items():
                grid=val_grid(torch.from_numpy(z),y)
                for (mode,rate),value in zip(refs,grid):
                    summary.append(dict(split=split,arm=name,mode=mode,rate=rate,macro_F1=value,matched_DPL=""))
                c=.1*counts(z[0]>0,labels)+.9*counts(z[1]>0,labels)
                sufficient[name]=aggregate(rows,c[:,None])
        else:
            masks=torch.load(CACHED/"test_predictions.pt",map_location="cpu")["masks"]
            for name,z in scores.items():
                binary=z>0
                c90=[]
                for mode in MODES:
                    for rate in RATES:
                        values=[]
                        for seed in SEEDS:
                            mask=masks[mode,rate,seed].numpy()[rows]
                            c=counts(binary[mask,np.arange(len(rows))],labels)
                            summed=c.sum(0)
                            value=200*(summed[0]/np.maximum(summed[1],1)).mean()
                            per_seed.append(dict(arm=name,mode=mode,rate=rate,seed=seed,macro_F1=value))
                            values.append(value)
                            if mode=="text" and rate==90:c90.append(c)
                        value=np.mean(values)
                        if name=="base":assert abs(value-float(refs[mode,rate]["selected"]))<1e-4
                        summary.append(dict(split=split,arm=name,mode=mode,rate=rate,macro_F1=value,
                                            matched_DPL=float(refs[mode,rate]["DPL"])))
                sufficient[name]=aggregate(rows,np.stack(c90,axis=1))
        intervals.extend(paired(split,sufficient))
    table("unseen_movie_ranking.csv",ranking_rows)
    table("nine_scenarios.csv",summary)
    table("paired_text90.csv",intervals)
    table("test_masks.csv",per_seed)
    if selection:
        result={}
        for s in selection:
            scored=[r for r in summary if r["split"]=="test" and r["arm"]==s["arm"]+"_selected"]
            result[s["arm"]]=dict(selected_epoch=s["selected"],trained_epoch=s["trained"],
                trainable=s["trainable"],epochs=s["epochs"],test_mean_F1=np.mean([r["macro_F1"] for r in scored]),
                scenarios_above_DPL=int(sum(r["macro_F1"]>r["matched_DPL"] for r in scored)),
                inference_parameters_added=0)
        (JOB/"results.json").write_text(json.dumps(result,indent=2))


if __name__=="__main__":
    torch.set_num_threads(2)
    report()
