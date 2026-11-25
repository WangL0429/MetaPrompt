import csv
import json
from pathlib import Path
from statistics import mean

job=Path(__file__).resolve().parent
visual=job.parent/'mmimdb_visual_prompt_NOSgpO'
cross=job.parent/'mmimdb_cross_meta_k9dMAw'
conditions=[(m,r) for m in ('both','image','text') for r in (.5,.7,.9)]
seeds=(42,2026,3407)
def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def grid(path, ck):
    rows=load_csv(path)
    assert len(rows)==27
    assert {Path(r['checkpoint']).resolve() for r in rows}=={ck.resolve()}
    scores={(r['missing_type'],float(r['missing_rate']),int(r['mask_seed'])):
            float(r['macro_f1_percent']) for r in rows}
    return {k:mean(scores[k[0],k[1],s] for s in seeds) for k in conditions}
def write(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
x=json.loads((job/'selection.json').read_text())
assert x['epochs']==20 and x['plots']==32278 and x['movies']==15552
assert [h['epoch'] for h in x['history']]==list(range(21))
baseline=x['history'][0]['grid']
hist=[]
for h in x['history']:
    g=h['grid']
    d=min(a-b for a,b in zip(g,baseline))
    assert h['eligible']==(d>=-.25)
    if h['epoch']:
        assert h['plots']==x['plots'] and h['steps']==x['steps_per_epoch']
        assert abs(sum(h['mean_weights'])-1)<1e-4 and h['mean_relative_curvature']>0
    row=dict(epoch=h['epoch'],eligible=h['eligible'],selected=h['epoch']==x['best_epoch'],
             worst_delta_pp=d,worst_f1=min(g),mean_f1=mean(g))
    row.update({f'{k}_{r:.1f}':v for (k,r),v in zip(conditions,g)})
    hist.append(row)
best=max((h for h in x['history'] if h['eligible']),key=lambda h:(min(h['grid']),sum(h['grid'])))
assert best['epoch']==x['best_epoch']
write(job/'validation.csv',hist)
old={(r['missing_type'],float(r['missing_rate'])):r for r in load_csv(cross/'comparison.csv')}
methods={'initial':grid(visual/'initial.csv',visual/'initial.ckpt'),
         'last':grid(job/'last.csv',job/'last.ckpt'),
         'selected':grid(job/'selected.csv',job/'selected.ckpt')}
rows=[]
for k in conditions:
    row=dict(missing_type=k[0],missing_rate=k[1],DPL=float(old[k]['DPL_specialist']),
             learned_128=float(old[k]['learned']))
    row.update({m:v[k] for m,v in methods.items()})
    row.update(last_minus_initial=row['last']-row['initial'],selected_minus_initial=row['selected']-row['initial'],
               selected_minus_DPL=row['selected']-row['DPL'])
    rows.append(row)
write(job/'comparison.csv',rows)
summary=dict(selected_epoch=x['best_epoch'],steps=x['steps_per_epoch']*20,
             admissible_epochs=[r['epoch'] for r in hist if r['eligible']])
for m in ('initial','learned_128','last','selected'):
    summary[m]=dict(mean=mean(r[m] for r in rows),wins=sum(r[m]>r['DPL'] for r in rows),
                    worst_gap=min(r[m]-r['DPL'] for r in rows))
(job/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
