"""Summarize paired test results for two new seeds and the seed-0 reference."""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_run(s, path):
    selection=json.loads((path/"selection.json").read_text())
    results=json.loads((path/"results.json").read_text())
    if results["seed"]!=s or results["n"]!=1000 or selection["updated_block"]!="text":
        raise ValueError(f"Unexpected protocol in {path}")
    with (path/"test_grid.csv").open() as stream:
        grid=list(csv.DictReader(stream))
    with (path/"per_mask.csv").open() as stream:
        masks=list(csv.DictReader(stream))
    if len(grid)!=18 or len(masks)!=54:
        raise ValueError(f"Incomplete test grid in {path}")
    rows=[]
    for metric in ("auroc","acc"):
        cases=[r for r in grid if r["metric"]==metric]
        for r in cases:
            matched=[m for m in masks if (m["metric"],m["mode"],m["rate"])==(metric,r["mode"],r["rate"])]
            assert len(matched)==3 and {int(m["mask"]) for m in matched}=={42,2026,3407}
            for k in ("candidate","ranked","delta"):
                assert abs(statistics.mean(float(m[k]) for m in matched)-float(r[k]))<1e-7
        cases.append(dict(mode="mean9",rate="",**results["mean9"][metric]))
        for r in cases:
            baseline,ranked=float(r["candidate"]),float(r["ranked"])
            rows.append(dict(seed=s,step=selection["selected"]["step"],metric=metric,mode=r["mode"],rate=r["rate"],
                             baseline=baseline,ranked=ranked,delta=ranked-baseline))
    return rows


def write(path,rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    rows=read_run(0,ROOT.parent/"hatememes_rank_text_uLdKlk")
    for seed in (1,2):
        rows.extend(read_run(seed,ROOT/f"seed{seed}/rank"))
    write(ROOT/"per_training_seed.csv",rows)
    means=[]
    for seeds in ((1,2),(0,1,2)):
        keys=dict.fromkeys((r["metric"],r["mode"],r["rate"]) for r in rows)
        for metric,mode,rate in keys:
            subset=[r for r in rows if r["seed"] in seeds and (r["metric"],r["mode"],r["rate"])==(metric,mode,rate)]
            assert len(subset)==len(seeds)
            means.append(dict(seeds=",".join(map(str,seeds)),metric=metric,mode=mode,rate=rate,
                              **{k:statistics.mean(r[k] for r in subset) for k in ("baseline","ranked","delta")},
                              delta_sample_sd=statistics.stdev(r["delta"] for r in subset),
                              positive_seeds=sum(r["delta"]>0 for r in subset)))
    write(ROOT/"seed_means.csv",means)


if __name__=="__main__":
    main()
