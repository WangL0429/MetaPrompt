"""Paired second-order training for the four preregistered missingness distributions."""
import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.flop_counter import FlopCounterMode

from protocol import ARMS, episode_masks, state_fractions, training_rate

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'mmimdb_joint_full_9aNxqw'
sys.path.insert(0, str(BASE))
import train as base
from model_io import INITIAL, load, outputs, plots, validate, validation_data, views
from rule import curvature, flat, unit

MICRO = 16
WORLD = 2


def selection_key(arm, grid, initial):
    if not all(map(math.isfinite, grid)):
        raise ValueError('Nonfinite validation result')
    if arm == 'high_text':
        return (grid[8],)
    if arm == 'high_image':
        return (grid[5],)
    return (min(grid), sum(grid)) if base.accepted(grid, initial) else None


def joint(net, names, params, data, idx, course_rate, mask_seed, rank, arm, digest):
    rng = np.random.default_rng([mask_seed, rank])
    support, router = data.extra(rng, idx)
    for indices in (idx, support, router):
        digest.update(indices.astype('<i8', copy=False).tobytes())
    sup = data.batch(support)
    route = data.batch(router)
    def masks(part, size):
        return [np.asarray(x, dtype=int) for x in episode_masks(arm, course_rate, mask_seed, rank, part, size)]
    sm = masks('support', len(support))
    rm = masks('router', len(router))
    mix = data.ref.new_tensor(state_fractions(arm, training_rate(arm, course_rate)))
    denom = (mix @ data.pred + data.prevalence).clamp_min(1e-5)
    stats = (mix @ data.tp / denom, denom)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        directions = torch.stack([unit(flat(base.support_loss(net, sup, m), params)) for m in sm])
    baseline = base.routing(net, data, router, rm, None, route, stats)
    changes = []
    for direction in directions:
        with torch.no_grad():
            candidate = {n: p-d for n,p,d in zip(names, params, views(base.ALPHA*direction, params))}
        changes.append(base.routing(net, data, router, rm, candidate, route, stats) - baseline)
    w = directions.new_tensor(base.choose_weights(torch.stack(changes).cpu().numpy()))
    delta = base.ALPHA * (w @ directions)
    fast = {n: (p-d).detach().requires_grad_() for n,p,d in zip(names, params, views(delta, params))}
    live_gradient = torch.zeros_like(delta)
    fast_gradient = torch.zeros_like(delta)
    counts = np.zeros(3, dtype='int64')
    for side, items in enumerate(np.array_split(idx, 2)):
        target, parameters, candidate = (live_gradient, params, None) if side == 0 else (fast_gradient, tuple(fast.values()), fast)
        mode, = masks(('live_query', 'fast_query')[side], len(items))
        counts += np.bincount(mode, minlength=3)
        for state in range(3):
            members = items[mode == state]
            for j in range(0, len(members), MICRO):
                sub = members[j:j+MICRO]
                img, tok, y = data.batch(sub)
                with torch.autocast('cuda', dtype=torch.float16):
                    z = outputs(net, img, tok, state, candidate)[0]
                loss = base.risk(z, y, data.ref[state, sub], data.pos, stats)
                target.add_(flat(loss * 128, parameters), alpha=len(sub)/(128*len(idx)))
    correction = torch.zeros_like(delta)
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        for wi,m in zip(w.tolist(), sm):
            correction.add_(curvature(base.support_loss(net, sup, m), params, fast_gradient), alpha=base.ALPHA*wi)
    gradient = live_gradient + fast_gradient - correction
    if not gradient.isfinite().all():
        raise FloatingPointError('Nonfinite second-order gradient')
    return gradient, w, correction.norm()/fast_gradient.norm().clamp_min(1e-12), counts


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('arm', choices=ARMS)
    arg.add_argument('seed', type=int, choices=(0,1))
    args = arg.parse_args()
    arm, seed = args.arm, args.seed
    out = HERE / f's{seed}_{arm}'
    out.mkdir(exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    assert int(os.environ['WORLD_SIZE']) == WORLD
    torch.cuda.set_device(rank)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get(f'CAP_RANK_{rank}', '0.38')))
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=45))
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    net, ck = load(INITIAL)
    net.cuda().eval().requires_grad_(False)
    net.model.prompt_learner.requires_grad_(True)
    names, params = zip(*net.model.prompt_learner.named_parameters())
    data = base.TrainData()
    size = len(data.rows)
    batches = math.ceil(size/(base.LOCAL_BATCH*WORLD))
    assert size == 32278 and len(data.ids) == 15552 and batches == 64
    schedule = base.make_schedule(batches*base.EPOCHS, seed, 'curriculum')
    opt = torch.optim.AdamW(params, lr=base.LR, weight_decay=0)
    def snapshot():
        return {n:p.detach().cpu().clone() for n,p in zip(names,params)}
    if rank == 0:
        ds = plots(ck['hyper_parameters']['config'], 'dev')
        assert set(str(i).zfill(7) for i in data.ids).isdisjoint(str(i).zfill(7) for i in ds.table['image_id'].to_pylist())
        dev,y = validation_data(ds)
    progress = out/'resume.pt'
    spec = (arm,seed,MICRO,WORLD,batches,base.EPOCHS,str(INITIAL))
    if progress.exists():
        state=torch.load(progress,map_location='cpu')
        assert state['spec']==spec
        with torch.no_grad():
            for n,p in zip(names,params): p.copy_(state['current'][n])
        opt.load_state_dict(state['opt'])
        history,best,chosen=state['history'],state['best'],state['chosen']
        del state
    else:
        initial = validate(net,dev,y) if rank==0 else []
        history = [dict(epoch=0,grid=initial,eligible=True)]
        best = snapshot() if rank==0 else {}
        chosen = 0
    for epoch in range(len(history), base.EPOCHS+1):
        tic=time.monotonic()
        permutation=np.random.default_rng(1000+epoch+10000*seed).permutation(size)
        digest=hashlib.sha256()
        counts=np.zeros(3,dtype='int64')
        weight_sum=torch.zeros(3,device='cuda')
        curvature_sum=torch.zeros((),device='cuda')
        profile_batch=int(np.random.default_rng([7451,seed,epoch]).integers(batches))
        sample_flops=0
        for b,global_idx in enumerate(np.array_split(permutation,batches)):
            idx=np.array_split(global_idx,WORLD)[rank]
            step=(epoch-1)*batches+b
            rate,_,mask_seed=schedule[step]
            meter=FlopCounterMode(display=False) if b==profile_batch else nullcontext()
            with meter:
                g,w,ratio,observed=joint(net,names,params,data,idx,rate,mask_seed,rank,arm,digest)
            if b==profile_batch: sample_flops=meter.get_total_flops()
            g.mul_(len(idx)/len(global_idx))
            dist.all_reduce(g)
            lr=base.LR*(.2+.8*.5*(1+math.cos(math.pi*step/(len(schedule)-1))))
            opt.param_groups[0]['lr']=lr
            opt.zero_grad(set_to_none=True)
            for p,component in zip(params,views(g,params)): p.grad=component
            opt.step()
            counts+=observed
            weight_sum+=w
            curvature_sum+=ratio
        acc=torch.cat([torch.as_tensor(counts,device='cuda',dtype=torch.float64),weight_sum.double(),
                       curvature_sum.reshape(1).double(),torch.tensor([sample_flops],device='cuda',dtype=torch.float64)])
        dist.all_reduce(acc)
        ids=[None]*WORLD
        dist.all_gather_object(ids,digest.hexdigest())
        assert int(acc[:3].sum().item())==size and acc[-1].item()>0
        if rank==0:
            grid=validate(net,dev,y)
            key=selection_key(arm,grid,history[0]['grid'])
            history.append(dict(epoch=epoch,grid=grid,eligible=key is not None,query_counts=acc[:3].long().tolist(),
                                sample_order_digest=ids,profile_batch=profile_batch,measured_step_flops=int(acc[-1].item()),
                                mean_weights=(acc[3:6]/(WORLD*batches)).tolist(),
                                mean_relative_curvature=acc[6].item()/(WORLD*batches),seconds=time.monotonic()-tic))
            if key is not None and key>selection_key(arm,history[chosen]['grid'],history[0]['grid']):
                chosen,best=epoch,snapshot()
            base.save(dict(spec=spec,current=snapshot(),opt=opt.state_dict(),history=history,best=best,chosen=chosen),progress)
            if epoch==1:
                first=dict(arm=arm,seed=seed,**history[-1])
                (out/'first_epoch.json').write_text(json.dumps(first,indent=2))
                if seed==0 and arm=='course_shared': print('FIRST_EPOCH '+json.dumps(first),flush=True)
        dist.barrier()
    if rank==0:
        audit=json.loads((HERE/'mask_audit.json').read_text())[seed]
        assert np.sum([r['query_counts'] for r in history[1:]],axis=0).tolist()==audit['query_counts'][arm]
        info=dict(arm=arm,seed=seed,epochs=base.EPOCHS,steps=batches,selected_epoch=chosen,source=str(INITIAL),
                  trainable=sum(p.numel() for p in params),microbatch=MICRO,history=history)
        for tag,ep,values in [('last',base.EPOCHS,snapshot()),('selected',chosen,best)]:
            path=out/f'{tag}.ckpt'
            if tag=='selected' and ep in (0,base.EPOCHS):
                if not path.exists(): path.symlink_to(INITIAL if ep==0 else out/'last.ckpt')
            else:
                for n,v in values.items(): ck['state_dict']['model.prompt_learner.'+n]=v
                ck['matched_specialization']=dict(arm=arm,seed=seed,epoch=ep)
                base.save(ck,path)
        (out/'selection.json').write_text(json.dumps(info,indent=2))
    dist.destroy_process_group()

if __name__=='__main__': main()
