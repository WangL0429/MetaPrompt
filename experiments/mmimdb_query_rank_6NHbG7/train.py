import datetime
import json
import math
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

JOB = Path(__file__).resolve().parent
CACHE = JOB.parent / "mmimdb_joint_full_9aNxqw"
sys.path.append(str(CACHE))
from model_io import (INITIAL, VIS, load, plots, outputs, choose_weights,
                      validate, validation_data, views)
from clip.datamodules.missing_schedule import make_schedule
from clip.modules.query_ranking import query_ranking_loss
from rule import flat, unit, curvature

RANK_WEIGHT = .1
EPOCHS = 20
LOCAL_BATCH = 256
MICRO = 32
ALPHA = .03
LR = 1e-5
T = .5
SUPPORT, ROUTER = 8,12
MODES = ('text','image','both')


def save(obj,path):
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


class TrainData:
    def __init__(self):
        d=torch.load(CACHE/'train_cache.pt',map_location='cpu')
        self.ids=d['ids']
        self.rows=d['rows'].numpy()
        self.bytoken=d['tokens'].cuda()
        self.labels=d['labels'].cuda()
        self.ref=d['ref'].cuda()
        self.posters=np.load(CACHE/'posters.npy',mmap_mode='r')
        self.by_movie=[[] for _ in self.ids]
        for ix,row in enumerate(self.rows):
            self.by_movie[row].append(ix)
        y=self.labels[self.rows]
        count=y.sum(0)
        self.pos=((len(y)-count)/count.clamp_min(1)).sqrt().clamp(1,8)
        p=(self.ref/T).sigmoid()
        self.tp=(p*y).mean(1)
        self.pred=p.mean(1)
        self.prevalence=y.mean(0)

    def extra(self,rng,excluded):
        seen=set(self.rows[excluded].tolist())
        selected=[]
        while len(selected)<SUPPORT+ROUTER:
            row=int(rng.integers(len(self.by_movie)))
            if row not in seen:
                seen.add(row)
                selected.append(rng.choice(self.by_movie[row]))
        return np.array(selected[:SUPPORT]),np.array(selected[SUPPORT:])

    def batch(self,index):
        return torch.from_numpy(self.posters[self.rows[index]]).cuda(),self.bytoken[index],self.labels[self.rows[index]]

    def population_stats(self,r):
        mix=self.ref.new_tensor([1-r,r/2,r/2])
        denom=(mix@self.pred+self.prevalence).clamp_min(1e-5)
        return (mix@self.tp)/denom,denom


def masks(n,rate,rng):
    order=rng.permutation(n)
    count=round(n*rate)
    masks=[]
    for kind in MODES:
        nt={'text':count,'image':0,'both':(count+int(order[0]%2))//2}[kind]
        mask=np.zeros(n,dtype=int)
        mask[order[:nt]]=1
        mask[order[nt:count]]=2
        masks.append(mask)
    return masks


def tagged(x, idx):
    return x[idx] if x is not None else None


def support_loss(net,data,mask):
    image,tokens,y=data
    loss=0.
    for state in range(3):
        idx=np.flatnonzero(mask==state)
        if not len(idx):
            continue
        z,im,tx=outputs(net,image[idx],tokens[idx],state)
        loss=loss+len(idx)/len(y)*F.binary_cross_entropy_with_logits(z,y[idx])
        if state==0:
            im,tx=F.normalize(im),F.normalize(tx)
            center=F.normalize(.2*im.detach()+.8*tx.detach())
            loss=loss+F.mse_loss(tx,center)+2*F.mse_loss(im,center)
    return loss


def risk(logits,target,ref,pos,stats):
    # First-order expansion of minus Macro-F1 in predicted probabilities.
    # Global train-set counts keep rare genre weights defined in small batches.
    fraction,denom=stats
    dp=(logits/T).sigmoid()-(ref/T).sigmoid()
    metric=(-2*(target-fraction)*dp/denom).mean()
    cls=F.binary_cross_entropy_with_logits(logits,target,pos_weight=pos)
    return cls+.5*metric+.5*F.mse_loss(logits,ref)


@torch.no_grad()
def routing(net,data,idx,masks_,params,cache,stats):
    image,tok,y=cache
    predictions=torch.stack([outputs(net,image,tok,state,params)[0] for state in range(3)])
    ar=torch.arange(len(idx),device=y.device)
    return torch.stack([risk(predictions[m,ar],y,data.ref[m,idx],data.pos,stats) for m in masks_])



def rank_grad_for_group(net,data,members,state,candidate):
    """All same-state pairs participate even when the encoder uses microbatches."""
    chunks = []
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        for j in range(0,len(members),MICRO):
            img,tok,_=data.batch(members[j:j+MICRO])
            chunks.append(outputs(net,img,tok,state,candidate)[0])
    scores=torch.cat(chunks).detach().requires_grad_()
    labels=data.labels[data.rows[members]]
    loss=query_ranking_loss(scores,labels,task='multilabel')
    slope,=torch.autograd.grad(loss,scores)
    return loss.detach(),slope


def joint(net,names,params,data,idx,rate,rng):
    support,router=data.extra(rng,idx)
    support_data=data.batch(support)
    router_data=data.batch(router)
    sm=masks(len(support),rate,rng)
    rm=masks(len(router),rate,rng)
    stats=data.population_stats(rate)
    # Use identical attention arithmetic in both support passes.
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
        dirs=torch.stack([unit(flat(support_loss(net,support_data,m),params)) for m in sm])
    base=routing(net,data,router,rm,None,router_data,stats)
    matrices=[]
    for direction in dirs:
        with torch.no_grad():
            candidate={n:p-d for n,p,d in zip(names,params,views(ALPHA*direction,params))}
        matrices.append(routing(net,data,router,rm,candidate,router_data,stats)-base)
    w=dirs.new_tensor(choose_weights(torch.stack(matrices).cpu().numpy()))
    delta=ALPHA*(w@dirs)
    fast={n:(p-d).detach().requires_grad_() for n,p,d in zip(names,params,views(delta,params))}
    live_grad=torch.zeros_like(delta)
    fast_grad=torch.zeros_like(delta)
    rank_sum=delta.new_zeros(())
    missing_count=0
    # Every plot receives one out-of-support classification loss per epoch.
    # Randomly split between the actual deployment prompt and its virtual update.
    for side,items in enumerate(np.array_split(idx,2)):
        dest=live_grad if side==0 else fast_grad
        parameters=params if side==0 else tuple(fast.values())
        candidate=None if side==0 else fast
        mode=np.zeros(len(items),dtype=int)
        n=round(len(items)*rate)
        perm=rng.permutation(len(items))
        half=(n+int(rng.integers(2)))//2
        mode[perm[:half]]=1
        mode[perm[half:n]]=2
        for state in range(3):
            members=items[mode==state]
            slope=None
            if state and len(members):
                missing_count+=len(members)
                if len(members)>MICRO:
                    group_rank,slope=rank_grad_for_group(net,data,members,state,candidate)
                    rank_sum+=group_rank*len(members)
            for j in range(0,len(members),MICRO):
                sub=members[j:j+MICRO]
                img,tok,y=data.batch(sub)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=outputs(net,img,tok,state,candidate)[0]
                loss=risk(logits,y,data.ref[state,sub],data.pos,stats)
                if state:
                    if slope is None:
                        term=query_ranking_loss(logits,y,task='multilabel')
                        rank_sum+=term.detach()*len(sub)
                    else:
                        # Convert the group-average logit gradient to the microbatch
                        # normalization used below. The scalar is only a VJP surrogate.
                        term=(logits*slope[j:j+len(sub)]).sum()*(len(members)/len(sub))
                    loss=loss+RANK_WEIGHT*term
                dest.add_(flat(loss*128,parameters),alpha=len(sub)/(128*len(idx)))
    correction=torch.zeros_like(delta)
    with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_mem_efficient=False,enable_math=True):
        for wi,mask in zip(w.tolist(),sm):
            correction.add_(curvature(support_loss(net,support_data,mask),params,fast_grad),alpha=ALPHA*wi)
    grad=live_grad+fast_grad-correction
    if not grad.isfinite().all():
        raise FloatingPointError('Nonfinite shared meta gradient')
    return grad,w,correction.norm()/fast_grad.norm().clamp_min(1e-12),rank_sum,missing_count


def accepted(grid,base):
    return all(math.isfinite(v) and v>=b-.25 for v,b in zip(grid,base))


def main():
    rank=int(os.environ.get('LOCAL_RANK',0))
    world=int(os.environ.get('WORLD_SIZE',1))
    torch.cuda.set_device(rank)
    if world>1:
        dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=45))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    net,ck=load(INITIAL)
    net.cuda().eval().requires_grad_(False)
    net.model.prompt_learner.requires_grad_(True)
    names,params=zip(*net.model.prompt_learner.named_parameters())
    data=TrainData()
    cfg=ck['hyper_parameters']['config']
    if rank==0:
        devds=plots(cfg,'dev')
        assert set(data.ids).isdisjoint(str(i) for i in devds.table['image_id'].to_pylist())
        dev,y=validation_data(devds)
    size=len(data.rows)
    batches=math.ceil(size/(LOCAL_BATCH*world))
    sched=make_schedule(batches*EPOCHS,0,'curriculum')
    opt=torch.optim.AdamW(params,lr=LR,weight_decay=0)
    def snapshot():
        return {n:p.detach().cpu().clone() for n,p in zip(names,params)}
    def restore(values):
        with torch.no_grad():
            for n,p in zip(names,params):
                p.copy_(values[n])
    progress=JOB/'resume.pt'
    if progress.exists():
        state=torch.load(progress,map_location='cpu')
        assert state['world']==world and state['size']==size
        restore(state['current'])
        opt.load_state_dict(state['opt'])
        history=state['history']
        start=state['epoch']+1
        best=state['best']
        best_epoch=state['best_epoch']
        best_grid=state['best_grid']
        base=history[0]['grid']
    else:
        start=1
        base=validate(net,dev,y) if rank==0 else []
        best_epoch=0
        best=snapshot() if rank==0 else None
        best_grid=base
        history=[dict(epoch=0,grid=base,eligible=True)]
    for epoch in range(start,EPOCHS+1):
        tic=time.monotonic()
        perm=np.random.default_rng(1000+epoch).permutation(size)
        count=0
        rank_sum=torch.zeros((),device="cuda")
        missing_count=0
        ws=torch.zeros(3,device='cuda')
        ratios=torch.zeros((),device='cuda')
        for b,global_idx in enumerate(np.array_split(perm,batches)):
            idx=np.array_split(global_idx,world)[rank]
            step=(epoch-1)*batches+b
            rate,_,seed=sched[step]
            rng=np.random.default_rng([seed,rank])
            grad,w,ratio,rank_value,n_missing=joint(net,names,params,data,idx,rate,rng)
            grad.mul_(len(idx)/len(global_idx))
            if world>1:
                dist.all_reduce(grad)
            lr=LR*(.2+.8*.5*(1+math.cos(math.pi*step/(len(sched)-1))))
            opt.param_groups[0]['lr']=lr
            opt.zero_grad(set_to_none=True)
            for p,g in zip(params,views(grad,params)):
                p.grad=g
            opt.step()
            ws+=w
            rank_sum+=rank_value
            missing_count+=n_missing
            ratios+=ratio
            count+=len(idx)
        stat=torch.cat((ws,ratios.reshape(1),rank_sum.reshape(1),ws.new_tensor([missing_count,count])))
        if world>1:
            dist.all_reduce(stat)
        assert int(stat[-1].item())==size
        if rank==0:
            grid=validate(net,dev,y)
            ok=accepted(grid,base)
            history.append(dict(epoch=epoch,grid=grid,eligible=ok,plots=size,steps=batches,
                                mean_weights=(stat[:3]/(world*batches)).tolist(),
                                mean_relative_curvature=(stat[3]/(world*batches)).item(),
                                mean_query_rank_loss=(stat[4]/stat[5].clamp_min(1)).item(),
                                missing_query_items=int(stat[5].item()),seconds=time.monotonic()-tic))
            if ok and (min(grid),sum(grid))>(min(best_grid),sum(best_grid)):
                best_epoch,best_grid,best=epoch,grid,snapshot()
            save(dict(epoch=epoch,world=world,size=size,opt=opt.state_dict(),current=snapshot(),
                      best=best,best_epoch=best_epoch,best_grid=best_grid,history=history),progress)
            if epoch==1:
                (JOB/'first_epoch.json').write_text(json.dumps(history[-1],indent=2))
                print('FIRST_EPOCH '+json.dumps(history[-1]),flush=True)
        if world>1:
            dist.barrier()
    if rank==0:
        result=dict(query_rank_weight=RANK_WEIGHT,epochs=EPOCHS,steps_per_epoch=batches,plots=size,movies=len(data.ids),
                    best_epoch=best_epoch,history=history,seed=0,inner_norm=ALPHA,outer_lr=LR)
        for label,epoch,values in [('last',EPOCHS,snapshot()),('selected',best_epoch,best)]:
            path=JOB/f'{label}.ckpt'
            if label=='selected' and epoch in (0,EPOCHS):
                if not path.exists():
                    path.symlink_to(INITIAL if epoch==0 else JOB/'last.ckpt')
            else:
                for n,tensor in values.items():
                    ck['state_dict']['model.prompt_learner.'+n]=tensor
                ck['cross_second_meta']=dict(epoch=epoch,**result)
                save(ck,path)
        (JOB/'selection.json').write_text(json.dumps(result,indent=2))
    if world>1:
        dist.destroy_process_group()

if __name__=='__main__':
    main()
