"""Matched 9-scenario test scores, specialist comparisons and movie-paired uncertainty."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from semantic_io import JOB, EXP, SOURCE, SEEDS, metadata, csv_out, json_out, setup


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def per_movie(ids, counts):
    totals = np.zeros((int(ids.max())+1, *counts.shape[1:]), dtype=np.float64)
    np.add.at(totals, ids, counts)
    return totals


def sufficient(labels, pred):
    return np.stack((labels*pred, labels.astype(np.int16)+pred), axis=-2)


def macro(totals):
    return 200*(totals[...,0,:]/np.maximum(totals[...,1,:],1)).mean(axis=-1)


def specialist(mode, rate, seed, ids, mask):
    path = EXP/"mmimdb_curriculum_20261119/dpl_specialists/mmimdb"/f"{mode}_{rate}"/f"seed_{seed}_results.json"
    d = json.loads(path.read_text())
    assert (d["status"],d["method"],d["test_missing_type"],round(100*d["test_missing_rate"]),d["seed"]) == (
        "success","DPL",mode,rate,seed)
    content = Path(d["mask_file"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == d["mask_sha256"]
    entries = {r["id"]: int(r["text_missing"])+2*int(r["image_missing"])
               for r in map(json.loads,content.splitlines())}
    assert len(ids)==len(entries)==d["n_unique_ids"] and d["n_items"]==16120
    assert np.array_equal([entries[i] for i in ids], mask.numpy())
    return 100*d["metrics"]["f1_macro"]


def bootstrap(counts, scenario):
    keys = list(counts)
    n = len(counts["base"])
    tables = np.concatenate([counts[k].reshape(n,-1) for k in keys], axis=1)
    def summarize(mult):
        sums = (mult @ tables).reshape(len(mult),len(keys),len(SEEDS),2,-1)
        return macro(sums).mean(axis=-1)
    rng = np.random.default_rng(1027)
    point = summarize(np.ones((1,n)))[0]
    draws = np.concatenate([summarize(rng.multinomial(n,np.full(n,1/n),size=50).astype(float))
                            for _ in range(30)])
    pairs = [(k,"base") for k in keys if k!="base"] + [
        ("k4_mean9","k1_mean9"),("k4_text90","k1_text90")]
    result = []
    for a,b in pairs:
        ia,ib = keys.index(a),keys.index(b)
        lo,hi = np.percentile(draws[:,ia]-draws[:,ib], [2.5,97.5])
        result.append(dict(scenario=scenario,arm=a,reference=b,delta_pp=point[ia]-point[ib],
                           lower95=lo,upper95=hi,movies=n,draws=len(draws)))
    return result


def compare():
    setup()
    prev = torch.load(EXP/"mmimdb_joint_residual_BO9efZ/test_predictions.pt",map_location="cpu")
    archive = metadata()["test"]
    rows = prev["rows"].numpy()
    y = prev["y"].numpy()
    masks = prev["masks"]
    assert np.array_equal(archive["rows"], rows)
    frozen = torch.load(JOB/"features_test.pt",map_location="cpu")
    assert torch.equal(frozen["y"],prev["y"]) and torch.equal(frozen["base"].transpose(0,1),prev["outputs"]["base"])
    del frozen
    z = {"base": prev["outputs"]["base"].transpose(0,1).numpy()}
    sels = {}
    for k in (1,4):
        out = JOB/f"k{k}"
        sel = json.loads((out/"selection.json").read_text())
        diag = json.loads((out/"test_diagnostics.json").read_text())
        for rule,epoch in sel["picks"].items():
            name = f"k{k}_{rule}"
            z[name] = torch.load(out/f"test_epoch_{epoch}.pt",map_location="cpu").numpy()
            sels[name] = dict(epoch=epoch,trainable=sel["trainable"],cal_grid=sel["history"][epoch-1]["cal_grid"],
                              **diag[str(epoch)])
    scen = [(m,r) for m in ("both","image","text") for r in (50,70,90)]
    base_expected = {(r["missing_type"],round(100*float(r["missing_rate"])),int(r["mask_seed"])):
                     float(r["macro_f1_percent"]) for r in read_csv(SOURCE.parent/"test.csv")}
    dpl_expected = {(r["missing_type"],round(100*float(r["missing_rate"]))):float(r["DPL"])
                    for r in read_csv(SOURCE.parent/"test_comparison.csv")}
    per_seed, grid, intervals = [], [], []
    for mode,rate in scen:
        values = {name:[] for name in (*z,"DPL")}
        paired = {name:[] for name in z}
        for seed in SEEDS:
            mask = masks[(mode,rate,seed)]
            state = mask.numpy()[rows]
            dpl = specialist(mode,rate,seed,archive["ids"],mask)
            line = dict(mode=mode,rate=rate,seed=seed,DPL=dpl)
            values["DPL"].append(dpl)
            for name,scores in z.items():
                pred = scores[np.arange(len(rows)),state]>0
                c = sufficient(y,pred)
                line[name] = float(macro(c.sum(0)))
                values[name].append(line[name])
                if (mode,rate) in (("image",90),("text",90)):
                    paired[name].append(per_movie(rows,c))
            assert abs(line["base"]-base_expected[mode,rate,seed]) < 1e-4
            per_seed.append(line)
        row = dict(mode=mode, rate=rate, **{n:float(np.mean(v)) for n,v in values.items()})
        assert abs(row["DPL"]-dpl_expected[mode,rate])<1e-4
        grid.append(row)
        if paired["base"]:
            intervals += bootstrap({n:np.stack(v,1) for n,v in paired.items()}, f"{mode}{rate}")
    grid.append(dict(mode="average",rate="",**{n:float(np.mean([r[n] for r in grid])) for n in (*z,"DPL")}))
    diagnostics = []
    for name,scores in z.items():
        for state in (0,1,2):
            logits=scores[:,state]
            diagnostics.append(dict(arm=name, state=state, macro_AP=100*average_precision_score(y,logits,average="macro"),
                                    macro_F1=float(macro(sufficient(y,logits>0).sum(0)))))
    csv_out(JOB/"test_grid.csv",grid)
    csv_out(JOB/"per_seed.csv",per_seed)
    csv_out(JOB/"paired_ci.csv",intervals)
    csv_out(JOB/"pure_states.csv",diagnostics)
    json_out(JOB/"selected_models.json",sels)
    print(grid[-1],flush=True)


if __name__=="__main__":
    compare()
