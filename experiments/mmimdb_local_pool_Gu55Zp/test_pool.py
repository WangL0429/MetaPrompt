"""One native forward per state, identical archived plot masks, movie-paired inference."""
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from common_pool import JOB,SOURCE,MODES,RATES,SEEDS,load,load_test_masks,metadata,plots,setup,write
from pool import Capture,classify,load_pooled_checkpoint,pooled_logits,projectors

ARMS=('global','uniform','weighted')


def f1(tp,pred,actual):
    return (200*tp/np.maximum(pred+actual,1)).mean(axis=-1)


def cost_row(kind, state, count, observed, local_count=0, width=0, embed=0):
    # Each path computes one token pooling and local projection. The weighted
    # path adds a query projection and a token dot product of the same sizes.
    paths = 2 if kind == 'weighted' else 1
    exact = 2 * paths * width * (local_count + embed)
    return dict(method=kind,state=state,params=count,op_counter_FLOPs=observed,
                additional_matmul_FLOPs=exact)


def ci_high(predictions,y,rows,masks):
    starts=np.r_[0,np.flatnonzero(np.diff(rows))+1]
    assert np.array_equal(rows[starts],np.arange(7799))
    def total(data):
        return np.add.reduceat(data.astype(np.int32),starts,axis=0)
    actual=total(y)
    n=len(starts)
    mult=np.random.default_rng(1601).multinomial(n,np.full(n,1/n),size=1500).astype(np.float64)
    actual_boot=(mult@actual)[:,None,:]
    records=[]; values={}
    indices=np.arange(len(rows))
    for name,pred in predictions.items():
        combined=[]
        for mode in MODES:
            pieces=[]
            for seed in SEEDS:
                p=pred[masks[mode,90,seed],indices]
                pieces.append(np.stack((total(p*y),total(p)),axis=1))
            cs=np.stack(pieces,axis=1)
            point=f1(cs.sum(0)[:,0],cs.sum(0)[:,1],actual.sum(0)).mean()
            boot=(mult@cs.reshape(n,-1)).reshape(len(mult),3,2,23)
            b=f1(boot[:,:,0],boot[:,:,1],actual_boot).mean(1)
            values[name,mode]=np.r_[point,b]
            combined.append(values[name,mode])
        values[name,'mean_high']=np.mean(combined,axis=0)
    for m in (*MODES,'mean_high'):
        for a,b in [('uniform','global'),('weighted','global'),('weighted','uniform')]:
            diff=values[a,m]-values[b,m]
            lo,hi=np.percentile(diff[1:],[2.5,97.5])
            records.append(dict(condition=m,method=a,reference=b,delta_pp=diff[0],lo95=lo,hi95=hi))
    return records


@torch.inference_mode()
def main():
    setup()
    net,ck=load(SOURCE)
    cfg=ck['hyper_parameters']['config'];del ck
    net.cuda().eval().requires_grad_(False)
    pools={}; packaged={}
    reference=net.state_dict()
    for name in ARMS[1:]:
        model,pool=load_pooled_checkpoint(JOB/f'{name}.ckpt')
        state=model.state_dict()
        assert all(torch.equal(v,state[k]) for k,v in reference.items())
        assert set(state)-set(reference)=={'local_token_pool.'+k for k in pool.state_dict()}
        pools[name]=pool
        packaged[name]=model
    meta=metadata()['test']
    ds=plots(cfg,'test')
    rows=meta['rows']
    assert meta['ids']==[str(i).zfill(7) for i in ds.table['image_id'].to_pylist()]
    assert np.array_equal(rows,[ds.index_mapper[i][0] for i in range(len(ds))])
    y=meta['labels'][rows]
    ids=[meta['ids'][i] for i in rows]
    masks={k:m.numpy() for k,m in load_test_masks(ids,MODES,RATES).items()}
    chunks={k:[[],[],[]] for k in ARMS}
    costs=[]; offsets=0
    with torch.autocast('cuda',dtype=torch.float16):
        for img,text in DataLoader(ds,batch_size=64,num_workers=4,pin_memory=True):
            img=img.cuda()
            for state in range(3):
                batch=dict(image=[torch.zeros_like(img) if state==2 else img],
                           text=(['']*len(img) if state==1 else text,),missing_type=[state]*len(img))
                zs={}
                with Capture(net,{state}) as cap:
                    f=net.infer(batch)['cls_feats']
                    zs['global']=net.mmimdb_classifier(f).float()
                    if state:
                        x,valid=cap.local(state)
                        h=f.chunk(2,dim=1)[state-1]
                        proj=projectors(net)[state-1]
                        for k,p in pools.items():
                            zs[k]=classify(net.mmimdb_classifier,p(h,x,valid,proj),state)
                            if offsets==0:
                                with FlopCounterMode(display=False) as counter:
                                    p(h[:1],x[:1],valid[:1],proj)
                                costs.append(cost_row(k,state,sum(t.numel() for t in p.parameters()),
                                                      counter.get_total_flops(),x.shape[1],x.shape[2],proj.shape[1]))
                    else:
                        zs.update({k:zs['global'] for k in pools})
                        if offsets==0:
                            costs.extend(cost_row(k,0,sum(t.numel() for t in p.parameters()),0)
                                         for k,p in pools.items())
                if offsets==0:
                    for k,p in pools.items():
                        actual=pooled_logits(packaged[k],p,batch)
                        torch.testing.assert_close(actual,zs[k],atol=0,rtol=0)
                for k in ARMS:
                    assert torch.isfinite(zs[k]).all()
                    chunks[k][state].append(zs[k].cpu())
            if offsets==0:
                # Mixed native batches and a text-only row with no content tokens.
                pattern=[0,1,2,2,1,0]
                part=img[:len(pattern)].clone()
                part[[2,3]]=0
                words=[text[0],'',text[2],'','',text[5]]
                mixed=dict(image=[part],text=(words,),missing_type=pattern)
                for k,p in pools.items():
                    native=net.mmimdb_classifier(net.infer(mixed)['cls_feats']).float()
                    actual=pooled_logits(packaged[k],p,mixed)
                    torch.testing.assert_close(actual[[0,3,5]],native[[0,3,5]],atol=0,rtol=0)
                    assert actual.isfinite().all()
                del packaged
            offsets+=len(img)
    assert offsets==16120
    scores={k:torch.stack([torch.cat(v) for v in groups]) for k,groups in chunks.items()}
    prediction={k:(v.sigmoid()>.5).numpy() for k,v in scores.items()}
    for k in ARMS[1:]:
        assert torch.equal(scores[k][0],scores['global'][0])
    idx=np.arange(len(rows))
    summary=[]; raw=[]; pure=[]
    with (SOURCE.parent/'test.csv').open() as f:
        historical={(r['missing_type'],round(float(r['missing_rate'])*100),int(r['mask_seed'])):
                    float(r['macro_f1_percent']) for r in csv.DictReader(f)}
    with (SOURCE.parent/'test_comparison.csv').open() as f:
        specialists={(r['missing_type'],round(float(r['missing_rate'])*100)):float(r['DPL']) for r in csv.DictReader(f)}
    parity=[]
    for mode in MODES:
        for rate in RATES:
            values={}
            for k,pred in prediction.items():
                vals=[]
                for seed in SEEDS:
                    p=pred[masks[mode,rate,seed],idx]
                    score=f1((p*y).sum(0),p.sum(0),y.sum(0))
                    vals.append(score)
                    raw.append(dict(method=k,mode=mode,rate=rate,mask_seed=seed,macro_f1=score))
                    if k=='global':parity.append(abs(score-historical[mode,rate,seed]))
                values[k]=np.mean(vals)
            summary.append(dict(mode=mode,rate=rate,**values,DPL=specialists[mode,rate],
                                weighted_minus_global=values['weighted']-values['global'],
                                weighted_minus_uniform=values['weighted']-values['uniform']))
    assert max(parity)<.02,(max(parity),parity)
    for k,pred in prediction.items():
        for s,label in enumerate(('complete','poster_only','text_only')):
            p=pred[s]
            pure.append(dict(method=k,state=label,macro_f1=f1((p*y).sum(0),p.sum(0),y.sum(0)),
                             macro_AP=100*average_precision_score(y,scores[k][s].numpy(),average='macro')))
    ci=ci_high(prediction,y,rows,masks)
    write('test.csv',summary);write('per_mask.csv',raw);write('pure_states.csv',pure)
    write('paired_ci.csv',ci);write('cost.csv',costs)
    print('Baseline reproduction max error pp:',max(parity),flush=True)
    print(summary,flush=True)

if __name__=='__main__':main()
