import csv
import json
from pathlib import Path
from statistics import mean
import numpy as np

job=Path(__file__).resolve().parent
baseline=job.parent/"mmimdb_visual_prompt_NOSgpO"
methods=("equal","fomaml","learned")
def read_scores(path):
    with path.open() as f:
        raw=list(csv.DictReader(f))
    scores={(r["missing_type"],float(r["missing_rate"]),int(r["mask_seed"])):float(r["macro_f1_percent"]) for r in raw}
    if len(scores)!=27 or len(raw)!=27:
        raise ValueError(f"Incomplete or ambiguous scores in {path}")
    return scores
values={m:read_scores(job/f"{m}.csv") for m in methods}
values["initial"]=read_scores(baseline/"initial.csv")
spec={}
with (baseline/"comparison.csv").open() as f:
    for r in csv.DictReader(f):
        spec[r["type"],float(r["rate"])]=float(r["specialist"])
rows=[]
for (kind,rate),expert in spec.items():
    row={"missing_type":kind,"missing_rate":rate,"DPL_specialist":expert}
    for m,scores in values.items():
        row[m]=mean(scores[kind,rate,s] for s in (42,2026,3407))
        row[m+"_gap_pp"]=row[m]-expert
    rows.append(row)
assert len(rows)==9
with (job/"comparison.csv").open("w") as f:
    writer=csv.DictWriter(f,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
summary={}
for method in values:
    worst=min(rows,key=lambda r:r[method+"_gap_pp"])
    summary[method]=dict(mean=mean(r[method] for r in rows),worst_gap=worst[method+"_gap_pp"],
                         worst_condition=[worst["missing_type"],worst["missing_rate"]],
                         wins=sum(r[method]>r["DPL_specialist"] for r in rows))
meta=[json.loads((job/f"{m}_validation.json").read_text()) for m in methods]
assert len({r["manifest"] for r in meta})==1
summary["selected_steps"]={r["arm"]:r["best_step"] for r in meta}
updates=[json.loads(line) for line in (job/"learned_updates.jsonl").read_text().splitlines()]
weights=np.asarray([u["weights"] for u in updates])
assert len(weights)==128 and np.allclose(weights.sum(axis=1),1)
summary["average_weights_text_image_both"]=weights.mean(axis=0).tolist()
summary["learned_wins_vs_equal"]=sum(r["learned"]>r["equal"] for r in rows)
summary["learned_wins_vs_fomaml"]=sum(r["learned"]>r["fomaml"] for r in rows)
(job/"summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
