import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch

JOB = Path(__file__).resolve().parent
OLD = JOB.parent / 'mmimdb_cross_meta_k9dMAw'
sys.path.insert(0, str(OLD))
from train_meta import (INITIAL, LR, SIZE, STEPS, episode, gradients, load, plots,
                        validate, validation_data, views)
from anchor import eligible, soft_grid, worst_drop

OBSERVE = (8,32,64,112,128)
GROUP_SIZE = 96
GROUPS = 8
CHUNK = 16


def make_guard(ds, episodes):
    reserved = {m for entry in episodes for m in entry['movies']}
    ids = [str(v) for v in ds.table['image_id'].to_pylist()]
    options = {}
    for plot, (row, _) in ds.index_mapper.items():
        if ids[row] not in reserved:
            options.setdefault(row, []).append(plot)
    rng = random.Random(53)
    rows = rng.sample(sorted(options), GROUP_SIZE*GROUPS)
    assert len(set(ids[r] for r in rows)) == len(rows)
    movie_ids = [ids[row] for row in rows]
    assert reserved.isdisjoint(movie_ids)
    groups = []
    for first in range(0,len(rows),GROUP_SIZE):
        subset = rows[first:first+GROUP_SIZE]
        indices = [rng.choice(options[row]) for row in subset]
        images = torch.stack([ds.get_image(i)['image'][0] for i in indices])
        texts = [ds.get_text(i)['text'] for i in indices]
        labels = torch.tensor([ds.table['label'][r].as_py() for r in subset],
                              device='cuda',dtype=torch.float32)
        groups.append((images,texts,labels))
    return groups, hashlib.sha256(json.dumps(movie_ids).encode()).hexdigest()


def batch_logits(net, group, state, start):
    images, texts, _ = group
    imgs = images[start:start+CHUNK].cuda()
    sentences = texts[start:start+CHUNK]
    batch = dict(image=[torch.zeros_like(imgs) if state == 2 else imgs],
                 text=(['']*len(sentences) if state == 1 else sentences,),
                 missing_type=[state]*len(sentences))
    with torch.autocast('cuda',dtype=torch.float16):
        return net.mmimdb_classifier(net.infer(batch)['cls_feats']).float()


@torch.no_grad()
def all_logits(net, group):
    return torch.stack([torch.cat([batch_logits(net, group, s, i)
                                  for i in range(0,GROUP_SIZE,CHUNK)]) for s in range(3)])


def guard_gradient(net, params, group, ref):
    # Differentiate the full 96-movie score once. Replay its vector-Jacobian
    # product by microbatch, so Macro-F1 is never averaged across microbatches.
    logits = all_logits(net,group).requires_grad_()
    scores = soft_grid(logits,group[2])
    loss = worst_drop(ref,scores)
    weights, = torch.autograd.grad(loss, logits)
    vector = torch.zeros(sum(p.numel() for p in params),device=logits.device)
    if loss.item() > 0:
        for s in range(3):
            if not weights[s].any():
                continue
            for i in range(0,GROUP_SIZE,CHUNK):
                live = batch_logits(net,group,s,i)
                tangent = weights[s,i:i+CHUNK]
                vector.add_(gradients((live*tangent*128.).sum(),params), alpha=1/128.)
    return vector, loss.item(), int((ref-scores.detach()).argmax())


def fo_gradient(net, names, params, support, query):
    terms = []
    for s,q in zip(support,query):
        inner = gradients(net.combined_loss_fn(s,None),params)
        fast = {n:p-d for n,p,d in zip(names,params,views(.01*inner,params))}
        terms.append(gradients(net._mmimdb_bce_loss(q,params=fast),params))
    return torch.stack(terms).mean(0)


def train(arm):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    coef = {'control':0., 'anchor':1.}[arm]
    net,ck = load(INITIAL)
    net.cuda().eval().requires_grad_(False)
    net.model.prompt_learner.requires_grad_(True)
    net.current_tasks = ['mmimdb']
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
            fo = fo_gradient(net,params_names,params,s,q)
            guard,loss,worst = guard_gradient(net,params,groups[group_id],refs[group_id])
            if step == 1 and (loss != 0 or guard.any()):
                raise RuntimeError('Step zero reference and live predictions differ')
            if not (fo.isfinite().all() and guard.isfinite().all()):
                raise FloatingPointError(f'Nonfinite gradient at step {step}')
            outer = fo + coef*guard
            opt.zero_grad(set_to_none=True)
            for p,g in zip(params,views(outer,params)):
                p.grad = g
            opt.step()
            print(json.dumps(dict(step=step,group=group_id,penalty=loss,worst=worst,
                                  fo_norm=fo.norm().item(),guard_norm=guard.norm().item())),file=f)
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
    meta = dict(arm=arm,coefficient=coef,manifest=hashlib.sha256(raw).hexdigest(),
                guard_manifest=guard_digest,group_size=GROUP_SIZE,groups=GROUPS,
                validation=hist,best_step=best_step,steps=STEPS,seed=0)
    path = JOB/f'{arm}.ckpt'
    if best_params is None:
        path.symlink_to(INITIAL)
    else:
        for name,value in best_params.items():
            ck['state_dict']['model.'+name] = value
        ck['anchored_fomaml'] = meta
        torch.save(ck,path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
    (JOB/f'{arm}_validation.json').write_text(json.dumps(meta,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm',choices=('control','anchor'))
    train(parser.parse_args().arm)
