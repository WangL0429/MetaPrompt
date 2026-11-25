"""Frozen poster probe. Fit on train; choose regularization and thresholds on dev-cal only."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

JOB=Path(__file__).resolve().parent
LAMBDAS=(.03,.3,3.,30.)
def ap(z,y):
    assert (y.sum(0)>0).all()
    return float(100*average_precision_score(y.numpy(),z.numpy(),average='macro'))

def predict(x,w,b):
    w,b=w.cuda().half(),b.cuda().half()
    return torch.cat([F.linear(t.cuda().half(),w,b).float().cpu() for t in x.split(64)])

def fit(train,w0,b0,lam):
    dtype=torch.float64
    x,y=[train[k].to(device='cuda',dtype=dtype) for k in ('x','labels')]
    w0,b0=w0.cuda().double(),b0.cuda().double()
    weight=train['weight'].to(device='cuda',dtype=dtype).unsqueeze(1)
    weight=weight/weight.sum()
    mu=(weight*x).sum(0)
    std=((x-mu).square()*weight).sum(0).sqrt().clamp_min(1e-5)
    zx=(x-mu)/std
    pos=(y*weight).sum(0)
    pos_weight=((1-pos)/pos.clamp_min(1e-8)).sqrt().clamp(1,8)
    base=F.linear(x,w0,b0)
    delta=torch.zeros_like(w0,requires_grad=True)
    bias=torch.zeros_like(b0,requires_grad=True)
    opt=torch.optim.LBFGS([delta,bias],lr=1.,max_iter=180,tolerance_grad=1e-8,
                          tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(base+F.linear(zx,delta,bias),y,
                                                pos_weight=pos_weight,reduction='none')
        loss=(loss*weight).sum(0).mean()+.5*lam*delta.square().sum(1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    final=float(closure().detach())
    gradmax=max(delta.grad.abs().max().item(),bias.grad.abs().max().item())
    assert np.isfinite(final) and gradmax<1e-4, (lam,final,gradmax)
    with torch.no_grad():
        dw=delta/std
        w=w0+dw
        b=b0+bias-dw@mu
        torch.testing.assert_close(F.linear(x[:100],w,b),base[:100]+F.linear(zx[:100],delta,bias),rtol=1e-8,atol=1e-8)
    return w.float().cpu(),b.float().cpu(),gradmax

def calibrate(z,y,movie):
    # Choose the grid cut per genre and shrink low-support cuts toward the global cut.
    # Score plot-level F1; use independent movie counts as the shrinkage sample size.
    cuts=torch.arange(-4.,4.001,.1)
    pred=z[None,:,:]>cuts[:,None,None]
    tp=(pred & y.bool()[None]).sum(1)
    f1=2*tp/(pred.sum(1)+y.sum(0)).clamp_min(1)
    center=cuts[f1.mean(1).argmax()]
    per=cuts[f1.argmax(0)]
    _,idx=np.unique(movie.numpy(),return_index=True)
    n=y[idx].sum(0)
    return (n*per+30*center)/(n+30)

def main():
    torch.set_num_threads(2)
    d=torch.load(JOB/'features.pt',map_location='cpu')
    w0,b0=d['w'],d['b']
    c=d['dev']['cal']
    assert w0[:,:512].numel()+b0.numel()==11799
    orig=predict(c['x'][1],w0,b0)
    torch.testing.assert_close(orig,c['z'][1],atol=.01,rtol=.001)
    records=[]
    options=[]
    for lam in LAMBDAS:
        w,b,grad=fit(d['train'],w0[:,:512],b0,lam)
        z=predict(c['x'][1],torch.cat((w,w0[:,512:]),1),b)
        score=ap(z,c['y'])
        records.append(dict(ridge=lam,cal_poster_AP=score,max_grad=grad))
        options.append((score,lam,w,b,z))
    _,lam,w,b,z=max(options,key=lambda r:(r[0],r[1]))
    orig_cut=calibrate(c['z'][1],c['y'],c['movie'])
    new_cut=calibrate(z,c['y'],c['movie'])
    result=dict(w=w,b=b,original_cut=orig_cut,probe_cut=new_cut,lambda_selected=lam,
                selection_metric='poster_macro_AP_on_dev_cal',candidates=records)
    path=JOB/'poster_head.pt'
    torch.save(result,path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)
    print(json.dumps(records), 'selected',lam,flush=True)

if __name__=='__main__':
    main()
