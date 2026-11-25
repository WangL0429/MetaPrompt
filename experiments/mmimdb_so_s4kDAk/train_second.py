import argparse
import hashlib
import json
import random

import torch
from second import JOB, meta_gradient
from train_anchor import (INITIAL, LR, OLD, STEPS, OBSERVE, GROUP_SIZE, GROUPS,
                          load, plots, validation_data, make_guard, soft_grid,
                          all_logits, validate, episode, guard_gradient, views, eligible)


def train(arm):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    coef = {'control':0., 'anchor':1.}[arm]
    net,ck = load(INITIAL)
    net.cuda().eval().requires_grad_(False)
    net.model.prompt_learner.requires_grad_(True)
    net.current_tasks = ['mmimdb']
    net.hparams.config['meta_second_order'] = True
    params_names = net._prompt_param_names()
    params = [net.model.get_parameter(n) for n in params_names]
    assert len(params) == sum(p.requires_grad for p in net.parameters())
    raw = (OLD/'episodes.json').read_bytes()
    schedule = json.loads(raw)
    assert len(schedule) == STEPS
    cfg = ck['hyper_parameters']['config']
    ds,dev_ds = (plots(cfg, s) for s in ('train','dev'))
    assert set(ds.table['image_id'].to_pylist()).isdisjoint(dev_ds.table['image_id'].to_pylist())
    dev,labels = validation_data(dev_ds)
    groups, guard_digest = make_guard(ds,schedule)
    refs = [soft_grid(all_logits(net,g),g[2]).detach() for g in groups]
    baseline = validate(net,dev,labels)
    best = (min(baseline), sum(baseline)/9)
    best_step, best_params = 0, None
    hist = [dict(step=0,grid=baseline,eligible=True)]
    opt = torch.optim.AdamW(params,lr=LR,weight_decay=0.)
    group_order = []
    rng = random.Random(71)
    while len(group_order) < STEPS:
        group_order.extend(rng.sample(range(GROUPS),GROUPS))
    with (JOB/f'{arm}_updates.jsonl').open('w',buffering=1) as f:
        for step,(entry,group_id) in enumerate(zip(schedule,group_order),1):
            s,q = episode(ds,entry)
            meta = meta_gradient(net,params,s,q)
            guard,loss,worst = guard_gradient(net,params,groups[group_id],refs[group_id])
            if step == 1 and (loss != 0 or guard.any()):
                raise RuntimeError('Step zero reference and live predictions differ')
            if not (meta.isfinite().all() and guard.isfinite().all()):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            outer = meta + coef*guard
            opt.zero_grad(set_to_none=True)
            for p,g in zip(params,views(outer,params)):
                p.grad = g
            opt.step()
            print(json.dumps(dict(step=step,group=group_id,penalty=loss,worst=worst,
                                  meta_norm=meta.norm().item(),guard_norm=guard.norm().item())),file=f)
            if step in OBSERVE:
                grid = validate(net,dev,labels)
                ok = eligible(grid,baseline)
                score = (min(grid),sum(grid)/9)
                delta = min(a-b for a,b in zip(grid,baseline))
                hist.append(dict(step=step,grid=grid,eligible=ok,worst_delta_pp=delta))
                if ok and score > best:
                    best, best_step = score, step
                    best_params = {n:p.detach().cpu().clone() for n,p in zip(params_names,params)}
                if step == 64:
                    print(f'FIRST_EPOCH {arm} step=64 min_delta_pp={delta:.3f} eligible={ok}',flush=True)
    meta = dict(arm=arm,order=2,coefficient=coef,manifest=hashlib.sha256(raw).hexdigest(),
                guard_manifest=guard_digest,group_size=GROUP_SIZE,groups=GROUPS,
                validation=hist,best_step=best_step,steps=STEPS,seed=0)
    path = JOB/f'{arm}.ckpt'
    if best_params is None:
        path.symlink_to(INITIAL)
    else:
        for name,value in best_params.items():
            ck['state_dict']['model.'+name] = value
        ck['anchored_maml'] = meta
        ck['hyper_parameters']['config']['meta_second_order'] = True
        torch.save(ck,path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
    (JOB/f'{arm}_validation.json').write_text(json.dumps(meta,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm',choices=('control','anchor'))
    train(parser.parse_args().arm)
