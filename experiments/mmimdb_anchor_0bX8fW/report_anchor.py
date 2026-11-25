import csv
import json
from pathlib import Path
from statistics import mean

job = Path(__file__).resolve().parent
visual = job.parent/'mmimdb_visual_prompt_NOSgpO'
old = job.parent/'mmimdb_cross_meta_k9dMAw'
modes = ('both','image','text')
rates = (.5,.7,.9)
arms = ('control','anchor')
seeds = (42,2026,3407)

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def scores(path, checkpoint):
    rows = read(path)
    keys = {(r['missing_type'],float(r['missing_rate']),int(r['mask_seed'])):
            float(r['macro_f1_percent']) for r in rows}
    assert {Path(r['checkpoint']).resolve() for r in rows} == {checkpoint.resolve()}
    assert len(rows)==27 and set(keys)=={(m,r,s) for m in modes for r in rates for s in seeds}
    return {(m,r):mean(keys[m,r,s] for s in seeds) for m in modes for r in rates}

meta = {arm:json.loads((job/f'{arm}_validation.json').read_text()) for arm in arms}
for k in ('manifest','guard_manifest'):
    assert meta['control'][k] == meta['anchor'][k]
assert meta['control']['manifest'] == json.loads((old/'fomaml_validation.json').read_text())['manifest']
base = meta['control']['validation'][0]['grid']
assert max(abs(a-b) for a,b in zip(base,meta['anchor']['validation'][0]['grid'])) < .015
results = {'initial':scores(visual/'initial.csv',visual/'initial.ckpt')}
summary = {}
with (job/'validation.csv').open('w') as f:
    names = ['arm','step','eligible','selected','worst_delta_pp','worst_f1','mean_f1',
             *[f'{m}_{r:g}' for m in modes for r in rates]]
    writer = csv.writer(f)
    writer.writerow(names)
    for arm, info in meta.items():
        updates = [json.loads(line) for line in (job/f'{arm}_updates.jsonl').read_text().splitlines()]
        assert [u['step'] for u in updates] == list(range(1,129))
        assert any(u['guard_norm']>0 and u['penalty']>0 for u in updates)
        hist = info['validation']
        assert [h['step'] for h in hist] == [0,8,32,64,112,128]
        original = hist[0]['grid']
        for h in hist:
            grid = h['grid']
            delta = min(a-b for a,b in zip(grid,original))
            assert h['eligible'] == (delta >= -.25)
            writer.writerow([arm,h['step'],h['eligible'],h['step']==info['best_step'],
                             delta,min(grid),mean(grid),*grid])
        expected = max((h for h in hist if h['eligible']),key=lambda h:(min(h['grid']),mean(h['grid'])))
        assert info['best_step'] == expected['step']
        results[arm] = scores(job/f'{arm}.csv',job/f'{arm}.ckpt')
        summary[arm] = dict(selected_step=info['best_step'],
                            admissible_steps=[h['step'] for h in hist if h['eligible']],
                            penalty_active_steps=sum(u['penalty']>0 for u in updates),
                            last_worst_delta=hist[-1]['worst_delta_pp'])
experts = {(r['type'],float(r['rate'])):float(r['specialist']) for r in read(visual/'comparison.csv')}
assert len(experts)==9
with (job/'comparison.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['missing_type','missing_rate','DPL_specialist','initial','control','anchor',
                     'anchor_minus_initial_pp','anchor_minus_DPL_pp'])
    for key in experts:
        a = results['anchor'][key]
        writer.writerow([*key,experts[key],*[x[key] for x in results.values()],
                         a-results['initial'][key],a-experts[key]])
for name, vals in results.items():
    summary.setdefault(name,{}).update(mean=mean(vals.values()),
        wins=sum(vals[k]>experts[k] for k in experts),worst_DPL_gap=min(vals[k]-experts[k] for k in experts))
(job/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,ensure_ascii=False))
