"""Seed-0 calibration on the frozen ranked classifier. Lock the validation choice before test."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold

from evaluate import infer, load_model
from food_io import CASES, JOB, SingleLabelDataset, condition_masks, csv_out, grid, loader, setup
from rank_test import paired_intervals
from single_label import decision, measure

OUT = Path(__file__).resolve().parent
RANK = JOB.parent / "hatememes_rank_text_uLdKlk"
ARMS = ("ranked", "ranked_cal", "ranked_raw", "ranked_raw_cal")
STATES = ("complete", "image_only", "text_only")


@torch.inference_mode()
def collect(split):
    dataset = SingleLabelDataset(split)
    net, generator, _ = load_model(RANK / "selected.ckpt")
    head = torch.load(JOB / "hatememes_selected.ckpt", map_location="cpu")["hatememes_image_readout"]
    model = net, generator, (head["weight"].cuda(), head["bias"].cuda())
    plain, raw_image = [], []
    for images, tokens, _, _ in loader(dataset):
        images, tokens = images.cuda(), tokens.cuda()
        columns = []
        for state in range(3):
            modes = torch.full((len(tokens),), state, device="cuda")
            if state == 1:
                z_raw, z = infer(model, images, tokens, modes, compare=True)
                raw_image.append(decision(z_raw).cpu())
            else:
                z = infer(model, images, tokens, modes)
            columns.append(decision(z).cpu())
        plain.append(torch.stack(columns, 1))
    ranked = torch.cat(plain).numpy()
    raw = ranked.copy()
    raw[:,1] = torch.cat(raw_image).numpy()
    return dataset.ids, dataset.labels.numpy(), ranked, raw


def fit_platt(x, y):
    params = []
    for column in x.T:
        def loss(v):
            z = v[0] * column + v[1]
            err = expit(z) - y
            return np.mean(np.logaddexp(0, z) - y*z), np.array([err @ column / len(y), err.mean()])
        r = minimize(loss, [1.,0.], method="L-BFGS-B", jac=True,
                     bounds=[(1e-5,None),(None,None)], options={"ftol":1e-13,"gtol":1e-9})
        if not r.success:
            raise ValueError(r.message)
        params.append(r.x)
    # Put all states on the complete-input log-odds scale; leave that branch exactly unchanged.
    slope, bias = np.asarray(params).T
    assert slope[0] > 1e-4
    return np.stack((slope/slope[0], (bias-bias[0])/slope[0]), axis=1)


def apply(x, p):
    return x * p[:,0] + p[:,1]


def metrics(x, y, masks):
    z, labels = torch.from_numpy(x), torch.from_numpy(y)
    record = {}
    for metric in ("auroc","acc"):
        scores, _ = grid(z, labels, masks, metric)
        record.update({metric+"_"+k: v for k,v in scores.items()})
        record[metric+"_mean9"] = float(np.mean(list(scores.values())))
        for i, state in enumerate(STATES):
            record[metric+"_"+state] = measure(z[:,i], labels, metric)
    return record


def decompose(scores, y, masks, split):
    records = []
    for (mode,rate,seed), tensor in masks.items():
        states = tensor.numpy()
        for arm, x in scores.items():
            selected = x[np.arange(len(y)),states]
            total_pairs = int((y==1).sum()) * int((y==0).sum())
            summed = 0.
            for i in range(3):
                pos = selected[(states==i) & (y==1)]
                for j in range(3):
                    neg = np.sort(selected[(states==j) & (y==0)])
                    count = len(pos)*len(neg)
                    if count:
                        below = np.searchsorted(neg,pos,side="left")
                        upto = np.searchsorted(neg,pos,side="right")
                        correct = .5*(below+upto).sum()
                        contribution = 100*correct/total_pairs
                        auc = 100*correct/count
                    else:
                        contribution, auc = 0., None
                    summed += contribution
                    records.append(dict(split=split,mode=mode,rate=rate,mask_seed=seed,arm=arm,
                                        pos_state=STATES[i],neg_state=STATES[j],pairs=count,
                                        weight=count/total_pairs,auc=auc,contribution=contribution))
            check = measure(torch.from_numpy(selected),torch.from_numpy(y))
            assert abs(summed-check) < 1e-9
    csv_out(OUT / f"{split}_auc_pairs.csv",records)


def summarize(split, ids, y, ranked, raw, params):
    scores = dict(zip(ARMS,(ranked,apply(ranked,params["ranked"]),raw,apply(raw,params["ranked_raw"]))))
    masks = condition_masks(ids,split)
    summary = {arm: metrics(x,y,masks) for arm,x in scores.items()}
    for i in (0,2):
        for state in STATES:
            k = "auroc_"+state
            assert abs(summary[ARMS[i]][k] - summary[ARMS[i+1]][k]) < 1e-9
    csv_out(OUT / f"{split}_arms.csv",[dict(arm=k,**v) for k,v in summary.items()])
    decompose(scores,y,masks,split)
    return summary,scores,masks


def run_fit():
    ids, y, ranked, raw = collect("train")
    assert len(ids)==len(set(ids))==8500
    folds = list(StratifiedKFold(5,shuffle=True,random_state=0).split(ranked,y))
    train_masks = condition_masks(ids,"train")
    diagnostics, params = [], {}
    for arm, x in (("ranked",ranked),("ranked_raw",raw)):
        oof = np.full_like(x,np.nan)
        for train, holdout in folds:
            oof[holdout] = apply(x[holdout],fit_platt(x[train],y[train]))
        assert np.isfinite(oof).all()
        for variant, scores in (("uncalibrated",x),("calibrated",oof)):
            diagnostics.append(dict(arm=arm,variant=variant,**metrics(scores,y,train_masks)))
        params[arm] = fit_platt(x,y)
    csv_out(OUT / "train_oof.csv",diagnostics)
    vids, vy, vranked, vraw = collect("val")
    assert len(vids)==500 and set(vids).isdisjoint(ids)
    summary, _, _ = summarize("val",vids,vy,vranked,vraw,params)
    baseline = summary["ranked"]
    eligible = [k for k in ARMS if summary[k]["auroc_image_90"] >= baseline["auroc_image_90"]
                and summary[k]["auroc_text_90"] >= baseline["auroc_text_90"]]
    selected = max(eligible, key=lambda k: summary[k]["auroc_mean9"])
    record = dict(training_seed=0,rank_source=str(RANK/"selected.ckpt"),raw_head_source=str(JOB/"hatememes_selected.ckpt"),
                  scale_and_bias={k:v.tolist() for k,v in params.items()}, selected=selected, validation=summary,
                  rule="Maximize validation mean9 AUROC among arms preserving both 90% directions versus ranked. Ties in listed order.")
    (OUT/"selection.json").write_text(json.dumps(record,indent=2))
    print("Validation locked:",selected,flush=True)


def run_test():
    lock = json.loads((OUT/"selection.json").read_text())
    ids, y, ranked, raw = collect("test")
    assert len(ids)==1000
    params = {k:np.array(v) for k,v in lock["scale_and_bias"].items()}
    summary,scores,masks = summarize("test",ids,y,ranked,raw,params)
    previous = json.loads((RANK/"results.json").read_text())
    for metric in ("auroc","acc"):
        assert abs(summary["ranked"][metric+"_mean9"]-previous["mean9"][metric]["ranked"]) < .001
    ci = {}
    for name in ARMS[1:]:
        ci[name] = paired_intervals({"candidate":torch.from_numpy(ranked),"ranked":torch.from_numpy(scores[name])},
                                    torch.from_numpy(y),masks)
    (OUT/"test.json").write_text(json.dumps(dict(selected=lock["selected"], metrics=summary, paired95_vs_ranked=ci),indent=2))
    print("Test complete:",lock["selected"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["fit","test"])
    args = parser.parse_args()
    setup()
    (run_fit if args.stage=="fit" else run_test)()
