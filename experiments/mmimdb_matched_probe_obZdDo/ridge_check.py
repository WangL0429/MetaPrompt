from compare import JOB, fit, logits, ap, thresholds, fscore, counts, write
import pickle, csv
import numpy as np
import torch

torch.set_num_threads(2)
with (JOB/'metadata.pkl').open('rb') as f: meta=pickle.load(f)
with (JOB/'calibration.csv').open() as f: records=list(csv.DictReader(f))
old=torch.load(JOB/'linear_heads.pt',map_location='cpu')
train=meta['train']
dev=meta['dev']
cal=dev['cal']
wdev=np.bincount(dev['rows'])
rows=[]
for arm in ('shared','dpl'):
    with np.load(JOB/(arm+'.npz')) as fs:
        data=dict(x=torch.from_numpy(fs['train_x']),labels=torch.from_numpy(train['labels']),
                  weight=torch.from_numpy(np.bincount(train['rows'])))
        xdev=fs['dev_x']
    orig=old[arm]
    candidates=[(float(next(r['cal_macro_AP'] for r in records if r['arm']==arm and r['selected']=='True')),
                 orig['ridge'],logits(xdev,orig['w'],orig['b']))]
    for ridge in (30.,300.):
        w,b,grad=fit(data,torch.zeros(23,512),torch.zeros(23),ridge)
        z=logits(xdev,w,b)
        value=ap(dev['labels'][cal],z[cal],wdev[cal])
        candidates.append((value,ridge,z))
    selected=max(candidates,key=lambda c:(c[0],c[1]))[1]
    for value,ridge,z in candidates:
        cut=thresholds(dev['labels'][cal],z[cal],wdev[cal])
        pred=z[~cal].astype(float)>cut
        y,wt=dev['labels'][~cal],wdev[~cal]
        rows.append(dict(arm=arm,ridge=ridge,selected=ridge==selected,cal_AP=value,
                         validation_poster_F1=fscore(counts(y,pred,wt).sum(0)),validation_poster_AP=ap(y,z[~cal],wt)))
write('ridge_sensitivity.csv',rows)
print(rows)

