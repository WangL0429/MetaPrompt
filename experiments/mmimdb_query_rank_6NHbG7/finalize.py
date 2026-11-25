import csv,json
from pathlib import Path
from statistics import mean

job=Path(__file__).resolve().parent
exp=job.parent
def read(path):
    with path.open() as f: return list(csv.DictReader(f))
def keyed(rows):
    return {(r['missing_type'],float(r['missing_rate'])):r for r in rows}
def write(filename,rows):
    with (job/filename).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
sel=json.loads((job/'selection.json').read_text())
hist=sel['history']
assert sel['epochs']==20 and [h['epoch'] for h in hist]==list(range(21))
assert all(h['plots']==32278 and h['mean_relative_curvature']>0 and h['missing_query_items']>0 and h['mean_query_rank_loss']>0 for h in hist[1:])
assert all(h['eligible']==all(a>=b-.25 for a,b in zip(h['grid'],hist[0]['grid'])) for h in hist)
assert sel['best_epoch']==max((h for h in hist if h['eligible']),key=lambda h:(min(h['grid']),sum(h['grid'])))['epoch']
orig=keyed(read(exp/'mmimdb_joint_full_9aNxqw'/'comparison.csv'))
best=keyed(read(exp/'mmimdb_joint_head_UR2YBk'/'test_comparison.csv'))
raw=read(job/'selected.csv')
assert len(raw)==27 and {Path(r['checkpoint']).resolve() for r in raw}=={(job/'selected.ckpt').resolve()}
scores={(r['missing_type'],float(r['missing_rate']),int(r['mask_seed'])):float(r['macro_f1_percent']) for r in raw}
rows=[]
for k,r in orig.items():
    value=mean(scores[k[0],k[1],seed] for seed in (42,2026,3407))
    prior=float(best[k]['selected']); dpl=float(r['DPL']); control=float(r['selected'])
    rows.append(dict(missing_type=k[0],missing_rate=k[1],second_order=control,prior_best=prior,rank=value,DPL=dpl,
                     gain_vs_second=value-control,gain_vs_prior=value-prior,gap_vs_DPL=value-dpl))
write('comparison.csv',rows)
print(json.dumps(dict(epoch=sel['best_epoch'],avg=mean(r['rank'] for r in rows),dpl_wins=sum(r['gap_vs_DPL']>0 for r in rows))),flush=True)
