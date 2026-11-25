"""Verify selection, matching and raw scores; compare fixed and validation-selected checkpoints."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from protocol import ARMS
HERE=Path(__file__).resolve().parent
EXP=HERE.parent
sys.path.insert(0,str(EXP/'mmimdb_curriculum_20261119'))
from evaluate_mmimdb import MODES, RATES, SEEDS, load_test_masks

CONTRASTS=(('C-B','high_text','high_shared','text'), ('D-B','high_image','high_shared','image'),
           ('B-A','high_shared','course_shared','text'), ('B-A','high_shared','course_shared','image'))
VIEWS=('complete','poster_only','text_only')


def write(name, records):
    path=HERE/(name+'.csv')
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w') as f:
        w=csv.DictWriter(f,fieldnames=records[0]); w.writeheader(); w.writerows(records)
    tmp.replace(path)


def score(tp, predicted, actual):
    return (200*tp/np.maximum(predicted+actual,1)).mean(axis=-1)


def read_ref():
    with (EXP/'mmimdb_joint_head_UR2YBk'/'test_comparison.csv').open() as f:
        return {(r['missing_type'],round(100*float(r['missing_rate']))):
                (float(r['DPL']),float(r['selected'])) for r in csv.DictReader(f)}


def verify_training(run, arm, seed, audit):
    info=json.loads((run/'selection.json').read_text())
    h=info['history']
    assert (info['seed'],info['arm'],info['epochs'],info['steps'])==(seed,arm,20,64)
    assert [v['epoch'] for v in h]==list(range(21))
    assert all(len(v['grid'])==9 and np.isfinite(v['grid']).all() for v in h)
    for v in h:
        ok=(arm in ('high_text','high_image') or
            all(x>=b-.25 for x,b in zip(v['grid'],h[0]['grid'])))
        assert ok==v['eligible']
    def key(v):
        if arm=='high_text':return (v['grid'][8],)
        if arm=='high_image':return (v['grid'][5],)
        return (min(v['grid']),sum(v['grid']))
    expected=max((v for v in h if v['eligible']), key=key)
    assert expected['epoch']==info['selected_epoch']
    totals=np.sum([v['query_counts'] for v in h[1:]],axis=0)
    assert totals.tolist()==audit['query_counts'][arm]
    assert all(sum(v['query_counts'])==32278 and v['mean_relative_curvature']>0 and
               abs(sum(v['mean_weights'])-1)<1e-4 for v in h[1:])
    costs=dict(train_seed=seed,arm=arm,selected_epoch=expected['epoch'],steps=1280,
               trainable=info['trainable'],seconds=sum(v['seconds'] for v in h[1:]),
               estimated_training_PFLOPs=sum(v['measured_step_flops']*64 for v in h[1:])/1e15,
               profiled_steps=20,complete=int(totals[0]),text_missing=int(totals[1]),image_missing=int(totals[2]))
    return h,costs


def load_archive(run,tag,meta):
    with np.load(run/(tag+'.npz'),allow_pickle=False) as d:
        assert Path(d['checkpoint'].item())==(run/(tag+'.ckpt')).resolve(strict=True)
        for k in ('rows','ids','y'):
            if k not in meta: meta[k]=d[k]
            else: assert np.array_equal(meta[k],d[k])
        pred=np.unpackbits(d['predictions'],count=23,axis=-1).astype(bool)
        assert pred.shape==(3,16120,23)
        return pred,d['native'],d['pure']


def main():
    reference=read_ref()
    audit=json.loads((HERE/'mask_audit.json').read_text())
    meta={}; masks={}; grids=[]; pure=[]; effects=[]; durations=[]; curves=[]
    for seed in (0,1):
        by_arm={}; path_digests=None
        for arm in ARMS:
            run=HERE/f's{seed}_{arm}'
            h,cost=verify_training(run,arm,seed,audit[seed])
            durations.append(cost)
            digests=[v['sample_order_digest'] for v in h[1:]]
            if path_digests is not None: assert digests==path_digests
            path_digests=digests
            for v in h:
                row=dict(train_seed=seed,arm=arm,epoch=v['epoch'],selected=v['epoch']==cost['selected_epoch'],eligible=v['eligible'])
                row.update({f'{m}_{r}':x for (m,r),x in zip(((m,r) for m in MODES for r in RATES),v['grid'])})
                curves.append(row)
            for tag in ('last','selected'):
                pred,native,pure_scores=load_archive(run,tag,meta)
                y=meta['y']
                nplots=len(y)
                if not masks:
                    ids=meta['ids'][meta['rows']].tolist()
                    masks={k:v.numpy() for k,v in load_test_masks(ids,MODES,RATES).items()}
                indices=np.arange(nplots)
                for m_index,m in enumerate(MODES):
                    for r_index,r in enumerate(RATES):
                        scores=[]
                        for s in SEEDS:
                            classified=pred[masks[m,r,s],indices]
                            scores.append(float(score((classified*y).sum(0),classified.sum(0),y.sum(0))))
                        np.testing.assert_allclose(scores,native[m_index,r_index],atol=2e-5,rtol=0)
                        dpl,previous=reference[m,r]
                        grids.append(dict(train_seed=seed,arm=arm,checkpoint=tag,
                                          epoch=20 if tag=='last' else cost['selected_epoch'],mode=m,rate=r,
                                          mask42=scores[0],mask2026=scores[1],mask3407=scores[2],
                                          mean_f1=np.mean(scores),DPL=dpl,prior_shared=previous,
                                          gap_DPL=np.mean(scores)-dpl))
                for state,name in enumerate(VIEWS):
                    p=pred[state]
                    val=float(score((p*y).sum(0),p.sum(0),y.sum(0)))
                    assert abs(val-pure_scores[state])<2e-5
                    pure.append(dict(train_seed=seed,arm=arm,checkpoint=tag,state=name,macro_f1=val))
                by_arm[arm,tag]=pred
        starts=np.r_[0,1+np.flatnonzero(np.diff(meta['rows']))]
        assert np.array_equal(meta['rows'][starts],np.arange(7799))
        def movie_totals(matrix):
            return np.add.reduceat(matrix.astype(np.int32),starts,axis=0)
        y=meta['y']
        actual=movie_totals(y)
        # A bootstrap draw is reused across arms, masks and checkpoint policies.
        rng=np.random.default_rng(319)
        n=len(starts)
        draws=rng.multinomial(n,np.full(n,1/n),size=2000).astype(float)
        actual_boot=(draws@actual)[:,None,:]
        cache={}
        def estimate(arm,tag,mode):
            k=arm,tag,mode
            if k not in cache:
                movie_counts=[]
                for s in SEEDS:
                    p=by_arm[arm,tag][masks[mode,90,s],indices]
                    movie_counts.append(np.stack((movie_totals(p*y),movie_totals(p)),axis=1))
                c=np.stack(movie_counts,axis=1)
                tp,den=np.moveaxis(c.sum(0),1,0)
                point=float(score(tp,den,actual.sum(0)).mean())
                sampled=(draws@c.reshape(n,-1)).reshape(len(draws),3,2,23)
                boot=score(sampled[:,:,0],sampled[:,:,1],actual_boot).mean(1)
                cache[k]=(point,boot)
            return cache[k]
        for tag in ('last','selected'):
            for label,a,b,m in CONTRASTS:
                va,aa=estimate(a,tag,m)
                vb,bb=estimate(b,tag,m)
                lo,hi=np.percentile(aa-bb,[2.5,97.5])
                effects.append(dict(train_seed=seed,checkpoint=tag,contrast=label,mode=m,rate=90,
                                    arm_a=a,arm_b=b,score_a=va,score_b=vb,difference_pp=va-vb,
                                    paired_movie_lo95=lo,paired_movie_hi95=hi))
    assert len(grids)==144 and len(pure)==48 and len(effects)==16 and len(durations)==8
    write('test_grid',grids); write('pure_states',pure); write('paired_effects',effects)
    write('training_cost',durations); write('validation',curves)
    print(json.dumps([r for r in effects if r['checkpoint']=='last']),flush=True)

if __name__=='__main__': main()
