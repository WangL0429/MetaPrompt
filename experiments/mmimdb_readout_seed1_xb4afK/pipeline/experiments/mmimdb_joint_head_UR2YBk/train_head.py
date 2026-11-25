import csv
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from extract_complete import JOB, PROBE, REF
sys.path.insert(0,str(PROBE))
from fit_probe import ap, predict
from common import val_grid, load

MIXES=(.5,.75)
RIDGES=(.03,.3,3.)
STEPS=(.125,.25,.5,1.)
KEY='mmimdb_classifier.weight'

def write_csv(name,rows):
    with (JOB/name).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

def tensors(d,complete):
    calc=lambda x:x.cuda().double()
    train=d['train']
    w0,b0=calc(d['w']),calc(d['b'])
    rows=complete['rows'].cuda()
    xp=calc(train['x']); xc=calc(complete['x'])
    yp=calc(train['labels'])
    yc=yp[rows]
    freq=calc(train['weight'])
    freq=freq/freq.sum()
    # Scaling only. Subtracting the feature mean would implicitly add a bias.
    av=.5*((freq[:,None]*xp).sum(0)+xc[:,:512].mean(0))
    var=.5*((freq[:,None]*(xp-av).square()).sum(0)+(xc[:,:512]-av).square().mean(0))
    scale=var.sqrt().clamp_min(1e-5)
    rate=(freq[:,None]*yp).sum(0)
    pos=((1-rate)/rate.clamp_min(1e-8)).sqrt().clamp(1,8)
    pc=F.linear(xc,w0,b0)
    pp=F.linear(xp,w0[:,:512],b0)
    return (xp/scale,xc[:,:512]/scale,yp,yc,pp,pc,freq,pos,w0[:,:512],scale)

def fit(data,mix,ridge):
    xp,xc,yp,yc,bp,bc,freq,pos,orig,scale=data
    delta=torch.zeros_like(orig,requires_grad=True)
    opt=torch.optim.LBFGS([delta],lr=1.,max_iter=240,tolerance_grad=5e-8,
                          tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        dp=F.linear(xp,delta)
        dc=F.linear(xc,delta)
        loss_p=F.binary_cross_entropy_with_logits(bp+dp,yp,pos_weight=pos,reduction='none')
        loss_c=F.binary_cross_entropy_with_logits(bc+dc,yc,pos_weight=pos)
        loss=mix*(loss_p*freq[:,None]).sum(0).mean()+(1-mix)*loss_c
        loss=loss+.5*ridge*delta.square().sum(1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    obj=closure().item()
    grad=delta.grad.abs().max().item()
    assert math.isfinite(obj) and grad<1e-4,(mix,ridge,obj,grad)
    with torch.no_grad():
        change=delta/scale
        torch.testing.assert_close(F.linear(xp[:16]*scale,change),F.linear(xp[:16],delta))
    return change.float().cpu(),grad

def updated(base,change,step):
    w=base.clone()
    w[:,:512].add_(change,alpha=step)
    assert torch.equal(w[:,512:],base[:,512:])
    return w

def metrics(data,w,b):
    z=torch.stack([predict(x,w,b) for x in data['x']])
    assert torch.equal(z[2],data['z'][2]),'text-only predictions changed'
    grid=val_grid(z,data['y'])
    yp=data['y']
    only=z[1]>0
    pure=(200*(only*yp).sum(0)/((only.sum(0)+yp.sum(0)).clamp_min(1))).mean().item()
    return grid,pure

def row(name,mix,ridge,step,grid,pure,base):
    low=min(v-b for v,b in zip(grid,base))
    out=dict(candidate=name,poster_fraction=mix,ridge=ridge,update_scale=step,
             eligible=low>=-.25,worst_delta_pp=low,min_f1=min(grid),mean_f1=sum(grid)/9,pure_poster_f1=pure)
    for k,v in zip([f'{m}_{r}' for m in ('both','image','text') for r in (.5,.7,.9)],grid):
        out[k]=v
    return out

def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    d=torch.load(PROBE/'features.pt',map_location='cpu')
    complete=torch.load(JOB/'complete.pt',map_location='cpu')
    assert complete['source']==str(REF/'selected.ckpt')
    train=tensors(d,complete)
    cal=d['dev']['cal']
    chooser=[]
    choices=[]
    for mix in MIXES:
        group=[]
        for ridge in RIDGES:
            change,grad=fit(train,mix,ridge)
            w=updated(d['w'],change,1.)
            score=ap(predict(cal['x'][1],w,d['b']),cal['y'])
            chooser.append(dict(poster_fraction=mix,ridge=ridge,cal_poster_macro_AP=score,max_grad=grad))
            group.append((score,ridge,change))
        _,ridge,change=max(group,key=lambda t:(t[0],t[1]))
        choices.append((mix,ridge,change))
    write_csv('calibration.csv',chooser)
    dev=d['dev']['val']
    base,pure=metrics(dev,d['w'],d['b'])
    record=json.loads((REF/'selection.json').read_text())
    archived=record['history'][record['best_epoch']]['grid']
    assert max(abs(a-b) for a,b in zip(base,archived))<.015
    results=[row('original',0,0,0,base,pure,base)]
    best=results[0]
    best_w=d['w']
    for mix,ridge,change in choices:
        for step in STEPS:
            w=updated(d['w'],change,step)
            grid,pure=metrics(dev,w,d['b'])
            r=row(f'poster_{mix}_scale_{step}',mix,ridge,step,grid,pure,base)
            results.append(r)
            if r['eligible'] and (r['min_f1'],r['mean_f1'])>(best['min_f1'],best['mean_f1']):
                best=r
                best_w=w
    for r in results:
        r['selected']=r is best
    write_csv('validation.csv',results)
    src=REF/'selected.ckpt'
    dest=JOB/'selected.ckpt'
    if best is results[0]:
        dest.symlink_to(src)
    else:
        ck=torch.load(src,map_location='cpu')
        sd=ck['state_dict']
        assert torch.equal(sd[KEY],d['w']) and torch.equal(sd['mmimdb_classifier.bias'],d['b'])
        sd[KEY]=best_w.clone()
        assert torch.equal(sd[KEY][:,512:],d['w'][:,512:])
        assert (sd[KEY][:,:512]!=d['w'][:,:512]).any()
        ck['joint_visual_head']=dict(source=str(src),selected=best,trainable_coefficients=11776)
        torch.save(ck,dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').replace(dest)
    net,export=load(dest)
    orig=torch.load(src,map_location='cpu')['state_dict']
    out=export['state_dict']
    assert set(orig)==set(out) and all(torch.equal(orig[k],out[k]) for k in orig if k!=KEY)
    assert torch.equal(orig[KEY][:,512:],out[KEY][:,512:]) and torch.equal(out[KEY],best_w)
    check,_=metrics(dev,out[KEY],out['mmimdb_classifier.bias'])
    assert min(check)==best['min_f1'] and sum(check)/9==best['mean_f1']
    summary=dict(selected=best,source=str(src),train_movies=len(d['train']['ids']),train_plots=len(complete['rows']),
                 calibration_movies=int(cal['movie'].unique().numel()),validation_movies=int(dev['movie'].unique().numel()),
                 validation_plots=len(dev['y']),eligible_nonzero=sum(r['eligible'] for r in results[1:]),
                 changed_weights=int((out[KEY]!=orig[KEY]).sum()),added_parameters=0)
    (JOB/'selection.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
