"""Paired movie bootstrap. The prespecified primary endpoint is dev shared swap minus native AP."""
import csv
import pickle
from pathlib import Path
import numpy as np
from fit_heads import MATCH, JOB, ap, table


def bootstrap_inputs(y, logits, weight):
    rank = logits.argsort(axis=0, kind="stable")[::-1].T
    groups = []
    for col, order in enumerate(rank):
        sorted_scores = logits[order,col]
        ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(order)-1]
        groups.append((order, ends, weight[order], weight[order]*y[order,col]))
    return groups


def macro_ap(groups, sampling):
    result = np.zeros(len(sampling), dtype=np.float64)
    for order, ends, weight, positive in groups:
        draw = sampling[:,order]
        total = np.cumsum(draw*weight,axis=1)[:,ends]
        tp = np.cumsum(draw*positive,axis=1)[:,ends]
        hit = np.diff(tp, prepend=0, axis=1)
        precision = np.divide(tp, total, out=np.zeros(tp.shape), where=total>0)
        result += (hit*precision).sum(axis=1)/np.maximum(tp[:,-1],1)
    return result * (100/len(groups))


def main():
    with (MATCH / "metadata.pkl").open("rb") as stream:
        dev = pickle.load(stream)["dev"]
    y = dev["labels"][~dev["cal"]]
    weight = np.bincount(dev["rows"])[~dev["cal"]]
    scores = []
    predictions = {}
    for arm in ("shared","dpl"):
        with (JOB / (arm+"_scores.csv")).open() as stream:
            scores.extend(csv.DictReader(stream))
        with np.load(JOB / (arm+"_pred.npz")) as cached:
            for kind in ("native","swap","raw"):
                predictions[arm+"_"+kind] = cached["dev_"+kind]
    table(JOB / "scores.csv", scores)
    groups = {name: bootstrap_inputs(y,z,weight) for name,z in predictions.items()}
    n = len(y)
    rng = np.random.default_rng(117)
    ones = np.ones((1,n),dtype=int)
    points = {name: macro_ap(info,ones)[0] for name,info in groups.items()}
    for name,z in predictions.items():
        np.testing.assert_allclose(points[name], ap(y,z,weight), rtol=1e-12)
    pairs = [("shared_swap","shared_native"),("shared_raw","shared_native"),
             ("dpl_swap","dpl_native"),("dpl_raw","dpl_native"),
             ("dpl_raw","shared_raw"),("dpl_native","shared_native")]
    draws = {pair:[] for pair in pairs}
    for _ in range(60):
        sample = rng.multinomial(n,np.full(n,1/n),size=25)
        ap_by_arm = {name: macro_ap(info,sample) for name,info in groups.items()}
        for pair in pairs:
            a,b = pair
            draws[pair].extend(ap_by_arm[a]-ap_by_arm[b])
    intervals = []
    for (a,b), values in draws.items():
        low,high = np.percentile(values,[2.5,97.5])
        intervals.append(dict(first=a,second=b,delta_AP=points[a]-points[b],lower95=low,upper95=high,
                              repeats=len(values),movies=n))
    table(JOB / "paired_validation.csv",intervals)
    print((JOB / "scores.csv").read_text())
    print((JOB / "paired_validation.csv").read_text())


if __name__ == "__main__":
    main()

