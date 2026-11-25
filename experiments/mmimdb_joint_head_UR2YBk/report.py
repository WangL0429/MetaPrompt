import csv
import json
from statistics import mean
from pathlib import Path

job=Path(__file__).resolve().parent
ref=job.parent/'mmimdb_joint_full_9aNxqw'
seeds=(42,2026,3407)
conditions=[(m,r) for m in ('both','image','text') for r in (.5,.7,.9)]
def read_csv(p):
    with p.open() as f:return list(csv.DictReader(f))
def scores(p,model):
    raw=read_csv(p)
    assert len(raw)==27 and {Path(x['checkpoint']).resolve() for x in raw}=={model.resolve()}
    d={(r['missing_type'],float(r['missing_rate']),int(r['mask_seed'])):float(r['macro_f1_percent']) for r in raw}
    assert set(d)=={c+(s,) for c in conditions for s in seeds}
    return d
old=scores(ref/'selected.csv',ref/'selected.ckpt')
new=scores(job/'test.csv',job/'selected.ckpt')
expert={(r['missing_type'],float(r['missing_rate'])):float(r['DPL']) for r in read_csv(ref/'comparison.csv')}
selection=json.loads((job/'selection.json').read_text())
valid=read_csv(job/'validation.csv')
chosen=max([r for r in valid if r['eligible']=='True'],key=lambda r:(float(r['min_f1']),float(r['mean_f1'])))
assert chosen['candidate']==selection['selected']['candidate'] and sum(r['selected']=='True' for r in valid)==1
rows=[]
for c in conditions:
    orig=mean(old[c+(s,)] for s in seeds)
    actual=mean(new[c+(s,)] for s in seeds)
    rows.append(dict(missing_type=c[0],missing_rate=c[1],DPL=expert[c],baseline=orig,selected=actual,
                     delta_vs_baseline_pp=actual-orig,delta_vs_DPL_pp=actual-expert[c]))
with (job/'test_comparison.csv').open('w') as f:
    out=csv.DictWriter(f,fieldnames=rows[0].keys())
    out.writeheader();out.writerows(rows)
summary=dict(candidate=chosen['candidate'],eligible_nonzero=selection['eligible_nonzero'],
             mean_original=mean(r['baseline'] for r in rows),mean_selected=mean(r['selected'] for r in rows),
             wins_original=sum(r['baseline']>r['DPL'] for r in rows),
             wins_selected=sum(r['selected']>r['DPL'] for r in rows),
             conditions_improved=sum(r['selected']>r['baseline'] for r in rows),
             mask_mean_gains={s:mean(new[c+(s,)]-old[c+(s,)] for c in conditions) for s in seeds})
(job/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
