"""Fit the same plot-weighted linear probe on every frozen representation."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

JOB = Path(__file__).resolve().parent
MATCH = JOB.parent / "mmimdb_matched_probe_obZdDo"
sys.path.insert(0, str(MATCH))
from compare import ap, counts, fscore, logits, thresholds
from probe_fit import fit

RIDGES = (.0003, .003, .03, .3, 3., 30., 300.)


def table(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main(arm):
    with (MATCH / "metadata.pkl").open("rb") as stream:
        meta = pickle.load(stream)
    train, dev = meta["train"], meta["dev"]
    cal = dev["cal"]
    labels = dev["labels"][cal]
    weights = np.bincount(dev["rows"])[cal]
    scores, candidates, predictions, heads = [], [], {}, {}
    with np.load(JOB / (arm+"_features.npz")) as arrays:
        for kind in ("native", "swap", "raw"):
            x = torch.from_numpy(arrays["train_"+kind])
            data = dict(x=x, labels=torch.from_numpy(train["labels"]),
                        weight=torch.from_numpy(np.bincount(train["rows"])))
            dev_x = arrays["dev_"+kind]
            w0 = torch.zeros(labels.shape[1], x.shape[1])
            b0 = torch.zeros(labels.shape[1])
            fits = []
            for ridge in RIDGES:
                w, b, grad = fit(data, w0, b0, ridge)
                z = logits(dev_x[cal], w, b)
                score = ap(labels, z, weights)
                fits.append((score, ridge, w, b, z))
                candidates.append(dict(encoder=arm, features=kind, ridge=ridge, cal_AP=score,
                                       max_gradient=grad))
            best, ridge, w, b, z = max(fits, key=lambda r:(r[0],r[1]))
            cuts = thresholds(labels, z, weights)
            heads[kind] = dict(w=w,b=b,cuts=cuts,ridge=ridge)
            for split, subset in (("dev", ~cal), ("test", slice(None))):
                ds = meta[split]
                y = ds["labels"][subset]
                plot_weight = np.bincount(ds["rows"])[subset]
                vec = (dev_x if split=="dev" else arrays["test_"+kind])[subset]
                z = logits(vec, w, b)
                predictions[split+"_"+kind] = z
                f1 = fscore(counts(y, z.astype(float)>cuts, plot_weight).sum(0))
                scores.append(dict(encoder=arm, features=kind, dim=x.shape[1], ridge=ridge,
                                   split=split, macro_AP=ap(y,z,plot_weight), macro_F1=float(f1)))
            print(arm, kind, "selected", ridge, flush=True)
    torch.save(heads, JOB / (arm+"_heads.pt"))
    table(JOB / (arm+"_scores.csv"), scores)
    table(JOB / (arm+"_calibration.csv"), candidates)
    np.savez(JOB / (arm+"_pred.npz"), **predictions)


if __name__=="__main__":
    torch.set_num_threads(2)
    main(sys.argv[1])

