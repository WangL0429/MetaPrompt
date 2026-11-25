"""Movie-paired MMIMDb frozen posterior readout comparison."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

JOB=Path(__file__).resolve().parent
EXP=JOB.parent
from probe_fit import fit

RIDGES=(.0003,.003,.03,.3,3.)
SEEDS=(42,2026,3407)

def write(name,rows):
    with (JOB/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

def ap(y,z,weight):
    assert np.all(y.T@weight>0)
    return 100*average_precision_score(y,z,sample_weight=weight,average='macro')

def genre_f1(tp,n):
    return 200*np.divide(tp,np.maximum(n,1))

def counts(y,p,w):
    return np.stack([w[:,None]*(y*p),w[:,None]*(y.astype(int)+p)],axis=1)

def fscore(c):
    return genre_f1(c[...,0,:],c[...,1,:]).mean(axis=-1)

def thresholds(y,z,w):
    # Quantile cutpoints cover cosine scores and BCE logits on their own scales.
    # Set threshold separately for each genre using only calibration films.
    result=[]
    for j in range(y.shape[1]):
        choices=np.unique(np.r_[np.quantile(z[:,j],np.linspace(0,1,201)),0.,
                                 np.nextafter(float(z[:,j].min()),-np.inf)])
        pred=z[:,j,None].astype(np.float64)>choices
        true=w*y[:,j]
        scores=genre_f1(true@pred,pred.T@w+true.sum())
        indices=np.flatnonzero(scores==scores.max())
        result.append(choices[indices[np.abs(choices[indices]).argmin()]])
    return np.asarray(result)

def logits(x,w,b):
    return x.astype(np.float32)@w.numpy().T+b.numpy()

def calibration(meta,arm):
    with np.load(JOB/(arm+'.npz')) as features:
        x=features['train_x']
        train=meta['train']
        weights=np.bincount(train['rows'])
        assert len(weights)==len(x) and x.shape==(15552,512)
        data=dict(x=torch.from_numpy(x.copy()),labels=torch.from_numpy(train['labels']),
                  weight=torch.from_numpy(weights))
        dev=meta['dev']
        subset=dev['cal']
        y=dev['labels'][subset]
        wplot=np.bincount(dev['rows'])[subset]
        xc=features['dev_x'][subset]
        native=features['dev_z'][subset]
        candidates=[]; fits=[]
        for ridge in RIDGES:
            w,b,grad=fit(data,torch.zeros(23,512),torch.zeros(23),ridge)
            z=logits(xc,w,b)
            score=ap(y,z,wplot)
            candidates.append(dict(arm=arm,ridge=ridge,cal_macro_AP=score,max_gradient=grad))
            fits.append((score,ridge,w,b,z))
        _,ridge,w,b,z=max(fits,key=lambda a:(a[0],a[1]))
        cuts=thresholds(y,z,wplot)
        native_cuts=thresholds(y,native,wplot)
        for c in candidates: c['selected']=c['ridge']==ridge
        return dict(w=w,b=b,cuts=cuts,native_cuts=native_cuts,ridge=ridge),candidates

def evaluate(meta,arm,head):
    with np.load(JOB/(arm+'.npz')) as features:
        output=[]; all_counts={}; predictions={}
        for split,key in [('dev','validation'),('test','test')]:
            ds=meta[split]
            keep=~ds['cal'] if split=='dev' else np.ones(len(ds['ids']),dtype=bool)
            weight=np.bincount(ds['rows'])[keep]
            y=ds['labels'][keep]
            original=features[split+'_z'][keep]
            probe=logits(features[split+'_x'][keep],head['w'],head['b'])
            sets=dict(native=(original,features[split+'_pred'][keep]),
                      native_calibrated=(original,original.astype(float)>head['native_cuts']),
                      probe_zero=(probe,probe>0),
                      probe_calibrated=(probe,probe.astype(float)>head['cuts']))
            if split=='test':
                native_complete=features['complete_pred']
                indices=ds['complete_indices']
            for method,(z,pred) in sets.items():
                c=counts(y,pred,weight)
                score=dict(split=key,arm=arm,readout=method,poster_macro_F1=float(fscore(c.sum(0))),
                           poster_macro_AP=ap(y,z,weight),movie_equal_F1=float(fscore(counts(y,pred,np.ones_like(weight)).sum(0))))
                if split=='test':
                    vals=[]
                    for seed,mask in zip(SEEDS,ds['masks']):
                        is_full=mask[ds['rows']]==0
                        assert np.all(np.isin(np.flatnonzero(is_full),indices))
                        mixed=pred[ds['rows']].copy()
                        mixed[is_full]=native_complete[is_full[indices]]
                        val=float(fscore(counts(ds['labels'][ds['rows']],mixed,np.ones(len(mixed),int)).sum(0)))
                        score['text90_seed'+str(seed)]=val
                        vals.append(val)
                    score['text90_mean_F1']=float(np.mean(vals))
                    all_counts[method]=c
                output.append(score)
    return output,all_counts

def intervals(pairs):
    rng=np.random.default_rng(117)
    names=['native','native_calibrated','probe_zero','probe_calibrated']
    by_arm={a:np.concatenate([pairs[a][n].reshape(len(pairs[a][n]),-1) for n in names],axis=1).astype(float)
            for a in pairs}
    movies=len(by_arm['shared'])
    probs=np.full(movies,1/movies)
    def gaps(mult):
        scores={arm:fscore((mult@matrix).reshape(-1,len(names),2,23)) for arm,matrix in by_arm.items()}
        g=scores['dpl']-scores['shared']
        return np.column_stack((g,g[:,0]-g[:,3],g[:,1]-g[:,3]))
    point=gaps(np.ones((1,movies)))[0]
    samples=np.vstack([gaps(rng.multinomial(movies,probs,size=50).astype(float)) for _ in range(30)])
    bounds=np.percentile(samples,[2.5,97.5],axis=0).T
    labels=names+['gap_closed_native_to_cal_probe','gap_closed_cal_native_to_cal_probe']
    return [dict(measure=n,gap_pp=float(p),lower95=float(lo),upper95=float(hi),paired_movie_samples=1500)
            for n,p,(lo,hi) in zip(labels,point,bounds)]

def main():
    torch.set_num_threads(2)
    with (JOB/'metadata.pkl').open('rb') as f: meta=pickle.load(f)
    records=[]; heads={}
    for arm in ('shared','dpl'):
        heads[arm],candidates=calibration(meta,arm)
        records.extend(candidates)
    write('calibration.csv',records)
    torch.save(heads,JOB/'linear_heads.pt')
    print('Selected on calibration',[(a,h['ridge']) for a,h in heads.items()],flush=True)
    results=[]; pairs={}
    for arm in heads:
        out,pairs[arm]=evaluate(meta,arm,heads[arm])
        results.extend(out)
    for split in ('validation','test'):
        write(split+'.csv',[r for r in results if r['split']==split])
    ci=intervals(pairs)
    write('paired_gaps.csv',ci)
    with (EXP/'mmimdb_joint_head_UR2YBk'/'test.csv').open() as f:
        base={int(r['mask_seed']):float(r['macro_f1_percent']) for r in csv.DictReader(f)
              if r['missing_type']=='text' and float(r['missing_rate'])==.9}
    hist=EXP/'mmimdb_curriculum_20261119'/'dpl_specialists'/'mmimdb'/'text_90'
    ref={s:100*json.loads((hist/f'seed_{s}_results.json').read_text())['metrics']['f1_macro'] for s in SEEDS}
    checks=[]
    for row in results:
        if row['split']=='test' and row['readout']=='native':
            for seed,reference in (base if row['arm']=='shared' else ref).items():
                actual=row['text90_seed'+str(seed)]
                checks.append(dict(arm=row['arm'],seed=seed,archived=reference,replayed=actual,error_pp=actual-reference))
    write('parity.csv',checks)
    assert len(checks)==6
    for item in checks: assert abs(item['error_pp'])<.08,item
    print('Native parity max absolute error (pp)',max(abs(c['error_pp']) for c in checks),flush=True)
    print(json.dumps([r for r in results if r['split']=='test']),flush=True)
    print(json.dumps(ci),flush=True)

if __name__=='__main__': main()

