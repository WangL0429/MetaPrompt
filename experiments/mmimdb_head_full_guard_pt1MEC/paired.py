import json
import torch
from train_guard import (JOB,PROBE,REF,CACHE,COEF,KEY,ap,predict,val_grid,updated,fit,tensors,write_csv)
from experiment import CONFIG
from statistics import mean

def pure(z,y):
    pred=z>0
    return (200*(pred*y).sum(0)/(pred.sum(0)+y.sum(0)).clamp_min(1)).mean().item()

def score(part,w,b):
    ref=part['z']
    z=torch.stack([predict(x,w,b) for x in part['x']])
    assert torch.equal(z[2],ref[2])
    grid=val_grid(z,part['y'])
    return dict(poster_f1=pure(z[1],part['y']),full_f1=pure(z[0],part['y']),
                text90=grid[-1],image90=grid[5],mean_f1=mean(grid),
                full_rms=(z[0]-ref[0]).square().mean().sqrt().item(),
                poster_rms=(z[1]-ref[1]).square().mean().sqrt().item(),
                minimum_vs_start=min(a-b for a,b in zip(grid,val_grid(ref,part['y']))))

def main():
    assert COEF>0
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    d=torch.load(PROBE/'features.pt',map_location='cpu')
    complete=torch.load(CACHE/'complete.pt',map_location='cpu')
    chosen=json.loads((JOB/'selection.json').read_text())['selected']
    control=JOB.parent/CONFIG['control']
    prior_ckpt=torch.load(control/'selected.ckpt',map_location='cpu')
    new_ckpt=torch.load(JOB/'selected.ckpt',map_location='cpu')
    prior,new=[ck['state_dict'][KEY] for ck in (prior_ckpt,new_ckpt)]
    base=d['w']; b=d['b']
    cases={'source':base,'prior_shared':prior,'protected':new}
    steps={'source':0,'prior_shared':json.loads((control/'selection.json').read_text())['selected']['update_scale'],
           'protected':chosen['update_scale']}
    if chosen['update_scale']:
        train=tensors(d,complete)
        delta,_=fit(train,chosen['poster_fraction'],chosen['ridge'],coefficient=0)
        # Calibrate the comparison scale on a different set of movies from those scored below.
        cal=d['dev']['cal']
        target=pure(predict(cal['x'][1],new,b),cal['y'])
        def dist(t):
            w=updated(base,delta,t)
            achieved=pure(predict(cal['x'][1],w,b),cal['y'])
            return abs(achieved-target),abs(t-chosen['update_scale'])
        t=min([i/64 for i in range(65)],key=dist)
        cases['unprotected_cal_poster_f1_match']=updated(base,delta,t)
        steps['unprotected_cal_poster_f1_match']=t
        x=d['train']['x'].double()
        freq=d['train']['weight'].double()
        def rms(change):
            z=x@change.double().T
            return (z.square().mean(1)@freq/freq.sum()).sqrt().item()
        t=rms(new[:,:512]-base[:,:512])/rms(delta)
        cases['unprotected_train_poster_rms_match']=updated(base,delta,t)
        steps['unprotected_train_poster_rms_match']=t
    stats={name:score(d['dev']['val'],w,b) for name,w in cases.items()}
    write_csv('mechanism.csv',[dict(method=n,step=steps[n],**m) for n,m in stats.items()])
    a,c=stats['protected'],stats['prior_shared']
    trigger=(chosen['update_scale']>0 and a['text90']>c['text90'] and a['full_f1']>=c['full_f1'])
    result=dict(replicate=trigger,delta_90text=a['text90']-c['text90'],delta_full=a['full_f1']-c['full_f1'],
                delta_poster=a['poster_f1']-c['poster_f1'],delta_mean=a['mean_f1']-c['mean_f1'])
    (JOB/'decision.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    print(json.dumps(stats),flush=True)

if __name__=='__main__': main()
