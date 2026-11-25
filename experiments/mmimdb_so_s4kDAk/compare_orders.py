import csv
import json
from pathlib import Path
from statistics import mean

here = Path(__file__).resolve().parent
pairs = {'first':here.parent/'mmimdb_anchor_0bX8fW','second':here}
arms = ('control','anchor')
def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w') as f:
        out=csv.DictWriter(f,fieldnames=rows[0].keys())
        out.writeheader()
        out.writerows(rows)
validation = []
test_scores = {}
meta = {}
for order,folder in pairs.items():
    test_scores[order] = {(r['missing_type'],r['missing_rate']):r for r in read(folder/'comparison.csv')}
    for arm in arms:
        meta[order,arm]=json.loads((folder/f'{arm}_validation.json').read_text())
    validation.extend(dict(order=order,**r) for r in read(folder/'validation.csv'))
for arm in arms:
    a,b = (meta[k,arm] for k in pairs)
    assert b['order']==2
    for field in ('manifest','guard_manifest','steps','seed','coefficient','groups','group_size'):
        assert a[field]==b[field], field
    assert [h['step'] for h in a['validation']]==[h['step'] for h in b['validation']]
    assert max(abs(x-y) for x,y in zip(a['validation'][0]['grid'],b['validation'][0]['grid'])) < .02
rows = []
assert set(test_scores['first'])==set(test_scores['second'])
for key,base in test_scores['first'].items():
    other = test_scores['second'][key]
    assert (base['initial'],base['DPL_specialist'])==(other['initial'],other['DPL_specialist'])
    row=dict(missing_type=key[0],missing_rate=key[1],initial=base['initial'],DPL=base['DPL_specialist'])
    for arm in arms:
        row[f'first_{arm}']=base[arm]
        row[f'second_{arm}']=other[arm]
        row[f'{arm}_second_minus_first_pp']=float(other[arm])-float(base[arm])
    rows.append(row)
assert len(rows)==9 and len(validation)==24
write(here/'orders_test.csv',rows)
write(here/'orders_val.csv',validation)
print('Verified nine matched conditions and all validation checkpoints',flush=True)
