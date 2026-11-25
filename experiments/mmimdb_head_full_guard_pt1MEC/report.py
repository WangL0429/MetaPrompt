import json,csv
from pathlib import Path
from statistics import mean
job=Path(__file__).resolve().parent
cfg=json.loads((job/'run_config.json').read_text())
old=job.parent/cfg['control']
ref=job.parent/cfg['source']
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def scores(p,ck):
    r=read(p)
    assert len(r)==27 and {Path(x['checkpoint']).resolve() for x in r}=={ck.resolve()}
    g={(x['missing_type'],float(x['missing_rate']),int(x['mask_seed'])):float(x['macro_f1_percent']) for x in r}
    assert set(g)=={(m,t,s) for m in ('both','image','text') for t in (.5,.7,.9) for s in (42,2026,3407)}
    return g
assert (job/'test.exit').read_text().strip()=='0'
g0=scores(old/'test.csv',old/'selected.ckpt')
g1=scores(job/'test.csv',job/'selected.ckpt')
gr=scores(ref/'selected.csv',ref/'selected.ckpt')
dpl={(r['missing_type'],float(r['missing_rate'])):float(r['DPL']) for r in read(job.parent/'mmimdb_joint_full_9aNxqw'/'comparison.csv')}
r=[]
for k,x in dpl.items():
    a,b,c=[mean(g[k+(s,)] for s in (42,2026,3407)) for g in (gr,g0,g1)]
    r.append(dict(mode=k[0],rate=k[1],source=a,previous=b,protected=c,DPL=x,gain_pp=c-b,gap_dpl_pp=c-x))
with (job/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=r[0].keys())
    w.writeheader()
    w.writerows(r)
s={}
for k in ('previous','protected'):
    s[k]=dict(mean=mean(a[k] for a in r),wins=sum(a[k]>a['DPL'] for a in r),worst_gap=min(a[k]-a['DPL'] for a in r))
s['wins_over_previous']=sum(a['gain_pp']>0 for a in r)
s['mean_gains_per_mask']={seed:mean(g1[k+(seed,)]-g0[k+(seed,)] for k in dpl) for seed in (42,2026,3407)}
(job/'summary.json').write_text(json.dumps(s,indent=2))
print(json.dumps(s),flush=True)
