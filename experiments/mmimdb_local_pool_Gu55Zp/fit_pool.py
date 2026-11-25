import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common_pool import JOB, SOURCE, load, metadata, setup, val_grid, write
from pool import LocalPool, classify, projectors
from clip.datamodules.missing_schedule import make_schedule

EPOCHS=20
BATCH=256
KINDS=('uniform','weighted')
NAMES=[f'{m}_{rate}' for m in ('both','image','text') for rate in (50,70,90)]
HIGH=(2,5,8)


def array(tmp,split,name):
    arr=np.load(tmp/f'{split}_{name}.npy',mmap_mode='c')
    return torch.from_numpy(arr).cuda()


class Features:
    def __init__(self,tmp,split):
        self.states={state:(array(tmp,split,f'h{state}'),array(tmp,split,f'x{state}'),
                            None if state==1 else array(tmp,split,'valid2')) for state in (1,2)}

    def predict(self,net,pool,state,idx):
        h,x,valid=self.states[state]
        tokens=x[idx]
        mask=(torch.ones(tokens.shape[:2],device=tokens.device,dtype=torch.bool)
              if valid is None else valid[idx])
        pooled=pool(h[idx],tokens,mask,projectors(net)[state-1])
        return classify(net.mmimdb_classifier,pooled,state)


def summarize(z,y):
    grid=val_grid(z,y)
    row={name:v for name,v in zip(NAMES,grid)}
    return dict(**row,worst_high=min(grid[i] for i in HIGH),mean_high=np.mean([grid[i] for i in HIGH]),mean_nine=np.mean(grid))


def main():
    setup()
    tmp=Path((JOB/'cache.path').read_text().strip())
    meta=metadata()
    train=Features(tmp,'train')
    dev=Features(tmp,'dev')
    rows=torch.tensor(meta['train']['rows'],device='cuda')
    y=torch.tensor(meta['train']['labels'],device='cuda',dtype=torch.float32)[rows]
    positive=y.sum(0)
    pos=((len(y)-positive)/positive.clamp_min(1)).sqrt().clamp(1,8)
    d=meta['dev']
    dy=torch.from_numpy(d['labels'][d['rows']]).long()
    cal=torch.from_numpy(d['cal'][d['rows']]).bool()
    assert len(np.unique(d['rows'][cal.numpy()]))==1444 and len(np.unique(d['rows'][~cal.numpy()]))==1164
    original=torch.stack([array(tmp,'dev',f'z{s}').cpu() for s in range(3)])
    net,ck=load(SOURCE)
    net.cuda().eval().requires_grad_(False)
    base=summarize(original[:,~cal],dy[~cal])
    archived=json.loads((SOURCE.parent/'selection.json').read_text())['selected']
    assert max(abs(base[name]-archived[name.rsplit('_',1)[0]+'_'+str(int(name.rsplit('_',1)[1])/100)]) for name in NAMES)<.02
    pools={kind:LocalPool(kind).cuda() for kind in KINDS}
    optim={k:torch.optim.Adam(p.parameters(),lr=.002) for k,p in pools.items()}
    counts={k:sum(p.numel() for p in module.parameters()) for k,module in pools.items()}
    assert counts=={'uniform':1,'weighted':513} and not any(p.requires_grad for p in net.parameters())
    dev_idx=torch.arange(len(dy),device='cuda')
    def scores(pool):
        with torch.no_grad():
            return torch.stack([original[0],*[torch.cat([dev.predict(net,pool,s,ids).cpu()
                               for ids in dev_idx.split(64)]) for s in (1,2)]])
    for module in pools.values():
        torch.testing.assert_close(scores(module),original,atol=0,rtol=0)
    batches=math.ceil(len(y)/BATCH)
    schedule=make_schedule(batches*EPOCHS,0,'curriculum')
    best={}; history=[]; exposures=np.zeros(3,dtype=int)
    for epoch in range(1,EPOCHS+1):
        order=np.random.default_rng(1000+epoch).permutation(len(y))
        for b,items in enumerate(np.array_split(order,batches)):
            rate,_,mask_seed=schedule[(epoch-1)*batches+b]
            rng=np.random.default_rng(mask_seed)
            n=round(len(items)*rate)
            image_count=(n+int(rng.integers(2)))//2
            missing=rng.permutation(items)[:n]
            exposures+=[len(items)-n,image_count,n-image_count]
            if n==0:continue
            idxs=[torch.as_tensor(chunk,device='cuda') for chunk in (missing[:image_count],missing[image_count:])]
            for kind,pool in pools.items():
                opt=optim[kind]
                opt.zero_grad(set_to_none=True)
                for state,idx in enumerate(idxs,1):
                    if not len(idx):continue
                    src=rows[idx] if state==1 else idx
                    z=train.predict(net,pool,state,src)
                    loss=F.binary_cross_entropy_with_logits(z,y[idx],pos_weight=pos)*len(idx)/len(items)
                    loss.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters()):
                    raise ValueError('Missing/nonfinite module gradient')
                opt.step()
        for kind,pool in pools.items():
            z=scores(pool)
            result=summarize(z[:,cal],dy[cal])
            gate=pool.gate.tanh().item()
            history.append(dict(method=kind,epoch=epoch,gate=gate,**result))
            key=(result['worst_high'],result['mean_nine'])
            if kind not in best or key>best[kind][0]:
                best[kind]=(key,epoch,{n:p.detach().cpu().clone() for n,p in pool.state_dict().items()})
        if epoch==1:
            print('FIRST_EPOCH',[(k,round(p.gate.tanh().item(),5)) for k,p in pools.items()],flush=True)
    write('calibration_history.csv',history)
    final=[dict(method='global',epoch=0,gate=0.,**base)]
    for kind,(_,epoch,params) in best.items():
        pool=pools[kind]
        pool.load_state_dict(params,strict=True)
        z=scores(pool)
        result=summarize(z[:,~cal],dy[~cal])
        final.append(dict(method=kind,epoch=epoch,gate=pool.gate.tanh().item(),**result))
        # Each exported checkpoint is self-contained and includes every original tensor.
        payload=dict(ck)
        state=dict(ck['state_dict'])
        state.update({'local_token_pool.'+n:t for n,t in params.items()})
        payload['state_dict']=state
        payload['local_token_pool']=dict(kind=kind,epoch=epoch,source=str(SOURCE),temperature=10.)
        path=JOB/(kind+'.ckpt')
        torch.save(payload,path.with_suffix('.tmp'));path.with_suffix('.tmp').replace(path)
        assert all(torch.equal(v,ck['state_dict'][k]) for k,v in ck['state_dict'].items())
    write('validation.csv',final)
    a,u,w=final
    passed=(w['mean_high']>max(u['mean_high'],a['mean_high']) and w['worst_high']>=a['worst_high']
            and min(w[k]-a[k] for k in NAMES)>=-.25)
    info=dict(epochs=EPOCHS,train_movies=len(meta['train']['ids']),train_plots=len(y),exposures=exposures.tolist(),
              selected={k:v[1] for k,v in best.items()},parameters=counts,weighted_validation_pass=bool(passed))
    (JOB/'selection.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()
