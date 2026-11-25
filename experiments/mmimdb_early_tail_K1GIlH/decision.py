import json
from pathlib import Path
job=Path(__file__).resolve().parent
baseline=job.parent/'mmimdb_joint_full_9aNxqw'
def selected(p):
    s=json.loads((p/'selection.json').read_text())
    return s['history'][s['best_epoch']]['grid'],s['best_epoch']
a,ae=selected(job)
b,be=selected(baseline)
d=[i-j for i,j in zip(a,b)]
result=dict(baseline_epoch=be,early_epoch=ae,validation_delta_pp=d,
            replicate=d[8]>0 and d[5]>=-.25 and min(d)>=-.25)
(job/'decision.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
