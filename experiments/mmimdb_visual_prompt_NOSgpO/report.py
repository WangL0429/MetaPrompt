import csv
import json
from pathlib import Path
from statistics import mean
import torch

job=Path(__file__).resolve().parent
arms=("initial","supervised","distilled")
def keyed(file):
    with file.open() as stream:
        return {(r["missing_type"],float(r["missing_rate"]),int(r["mask_seed"])):float(r["macro_f1_percent"])
                for r in csv.DictReader(stream)}
scores={a:keyed(job/f"{a}.csv") for a in arms}
assert all(len(v)==27 and set(v)==set(scores["initial"]) for v in scores.values())
parent=job.parent/"mmimdb_head_QeuxLr"
with (parent/"comparison.csv").open() as f:
    prev=list(csv.DictReader(f))
seed_list=(42,2026,3407)
rows=[]
for old in prev:
    mode,rate=old["missing_type"],float(old["missing_rate"])
    row={"type":mode,"rate":rate,"previous":float(old["optimized"]),"specialist":float(old["DPL_specialist"])}
    for a in arms:
        row[a]=mean(scores[a][mode,rate,s] for s in seed_list)
    rows.append(row)
with (job/"comparison.csv").open("w") as f:
    w=csv.DictWriter(f,fieldnames=rows[0])
    w.writeheader()
    w.writerows(rows)
orig=torch.load(job/"initial.ckpt",map_location="cpu")["state_dict"]
for a in arms[1:]:
    candidate=torch.load(job/f"{a}.ckpt",map_location="cpu")
    weights=candidate["state_dict"]
    assert weights.keys()==orig.keys()
    changed={n for n in orig if not torch.equal(orig[n],weights[n])}
    assert changed==set(candidate["visual_prompt_tuning"]["trainable_names"]), changed
    print(a, "changed:", changed,"selected epoch:",candidate["visual_prompt_tuning"]["epoch"])
print(json.dumps(rows))
