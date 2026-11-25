import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

JOB = Path(__file__).resolve().parent
VISUAL = JOB.parent / "mmimdb_visual_prompt_NOSgpO"
sys.path.insert(0, str(VISUAL))
from common import INITIAL, load, plots, split_movies, val_grid
from clip.datamodules.missing_schedule import make_schedule
from meta_weights import UNIFORM, choose_weights

TASKS = ("text", "image", "both")
SIZE = 20
STEPS = 128
PROBE_NORM = 0.03
LR = 1e-4


def prepare():
    import pyarrow as pa
    from features import DATA
    table = pa.ipc.RecordBatchFileReader(pa.memory_map(str(DATA / "mmimdb_train.arrow"),"r")).read_all()
    ids = table["image_id"].to_pylist()
    assert len(set(ids)) == len(ids)
    plots_per_movie = table["plots"].to_pylist()
    starts = np.cumsum([0] + [len(texts) for texts in plots_per_movie]).tolist()
    rng = random.Random(19)
    movies = rng.sample(range(len(ids)), 2*SIZE*STEPS)
    schedule = []
    for step, (rate, _, mask_seed) in enumerate(make_schedule(STEPS, 0, "curriculum")):
        members = movies[step*2*SIZE:(step+1)*2*SIZE]
        indices = [rng.randrange(starts[i], starts[i+1]) for i in members]
        schedule.append(dict(rate=rate, mask_seed=mask_seed, plots=indices,
                             movies=[str(ids[i]) for i in members]))
    (JOB/"episodes.json").write_text(json.dumps(schedule, separators=(",", ":")))


def scenarios(images, texts, labels, permutation, ratio):
    count = round(SIZE*ratio)
    batches = []
    for mode in TASKS:
        mask = torch.zeros(SIZE, dtype=torch.long)
        n_text = {"text":count, "image":0, "both":(count+1)//2}[mode]
        mask[permutation[:n_text]] = 1
        mask[permutation[n_text:count]] = 2
        masked_images = images.clone()
        masked_images[mask == 2] = 0
        sentences = ["" if state == 1 else text for state,text in zip(mask.tolist(),texts)]
        batches.append(dict(image=[masked_images],text=(sentences,),missing_type=mask.to(images.device),
                            label=labels))
    return batches


def episode(ds, spec):
    indices = spec["plots"]
    rows = [ds.index_mapper[i][0] for i in indices]
    assert [str(ds.table["image_id"][r].as_py()) for r in rows] == spec["movies"]
    assert len(set(rows)) == SIZE*2
    images = torch.stack([ds.get_image(i)["image"][0] for i in indices]).cuda()
    texts = [ds.get_text(i)["text"] for i in indices]
    labels = [ds.table["label"][r].as_py() for r in rows]
    rng = random.Random(spec["mask_seed"])
    sides = []
    for start in (0,SIZE):
        end = start+SIZE
        perm = rng.sample(range(SIZE),SIZE)
        sides.append(scenarios(images[start:end], texts[start:end], labels[start:end], perm,spec["rate"]))
    return sides


def gradients(loss, params):
    gradients = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).flatten()
                      for p,g in zip(params,gradients)])


def views(vector, parameters):
    return [v.view_as(p) for v,p in zip(vector.split([p.numel() for p in parameters]),parameters)]


def update(model, optimizer, names, parameters, support, query, arm):
    gradients_s = torch.stack([gradients(model.combined_loss_fn(s, None),parameters) for s in support])
    norms = gradients_s.norm(dim=1,keepdim=True).clamp_min(1e-12)
    directions = gradients_s / norms
    with torch.no_grad():
        base = torch.stack([model._mmimdb_bce_loss(q) for q in query])
    changes = []
    diagonal_gradients = []
    for j in range(3):
        step = .01*gradients_s[j] if arm == "fomaml" else PROBE_NORM*directions[j]
        fast = {n:p-d for n,p,d in zip(names,parameters,views(step,parameters))}
        losses = []
        for k,q in enumerate(query):
            with torch.set_grad_enabled(arm == "fomaml" and j == k):
                loss = model._mmimdb_bce_loss(q, params=fast)
            if arm == "fomaml" and j == k:
                diagonal_gradients.append(gradients(loss,parameters))
            losses.append(loss.detach())
        changes.append(torch.stack(losses)-base)
    cross = torch.stack(changes).cpu().numpy()
    weights = choose_weights(cross) if arm == "learned" else UNIFORM
    if arm == "fomaml":
        outer = torch.stack(diagonal_gradients).mean(0)
    else:
        outer = torch.as_tensor(weights,device=directions.device,dtype=directions.dtype) @ directions
        outer = outer*norms.mean()
    if not torch.isfinite(outer).all():
        raise FloatingPointError("Non-finite meta gradient")
    optimizer.zero_grad(set_to_none=True)
    for p,g in zip(parameters,views(outer,parameters)):
        p.grad = g
    optimizer.step()
    return weights.tolist(),cross.tolist()


def validation_data(ds):
    ids = [str(ds.table["image_id"][ds.index_mapper[i][0]].as_py()).zfill(7) for i in range(len(ds))]
    indices = (~split_movies(ids)).nonzero().flatten().tolist()
    labels = torch.tensor([ds.table["label"][ds.index_mapper[i][0]].as_py() for i in indices])
    return DataLoader(Subset(ds,indices),batch_size=64,num_workers=4,pin_memory=True),labels


@torch.inference_mode()
def validate(model, loader, labels):
    pieces = [[],[],[]]
    with torch.autocast("cuda",dtype=torch.float16):
        for images,texts in loader:
            images = images.cuda(non_blocking=True)
            for mt,chunks in enumerate(pieces):
                batch = dict(image=[torch.zeros_like(images) if mt==2 else images],
                             text=([""]*len(texts) if mt==1 else texts,),missing_type=[mt]*len(texts))
                chunks.append(model.mmimdb_classifier(model.infer(batch)["cls_feats"]).float().cpu())
    return val_grid(torch.stack([torch.cat(part) for part in pieces]),labels)


def train(arm):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    net, ck = load(INITIAL)
    net.cuda().eval().requires_grad_(False)
    net.model.prompt_learner.requires_grad_(True)
    net.current_tasks = ["mmimdb"]
    names = net._prompt_param_names()
    parameters = [net.model.get_parameter(n) for n in names]
    assert len(parameters) and sum(p.requires_grad for p in net.parameters()) == len(names)
    cfg = ck["hyper_parameters"]["config"]
    ds = plots(cfg,"train")
    dev_ds = plots(cfg,"dev")
    assert set(ds.table["image_id"].to_pylist()).isdisjoint(dev_ds.table["image_id"].to_pylist())
    dev,labels = validation_data(dev_ds)
    opt = torch.optim.AdamW(parameters,lr=LR,weight_decay=0.)
    protocol = (JOB/"episodes.json").read_bytes()
    schedule = json.loads(protocol)
    assert len(schedule)==STEPS
    history=[]
    best_score=None
    metadata=dict(arm=arm, manifest=hashlib.sha256(protocol).hexdigest(),seed=0,steps=STEPS,
                  support=SIZE,query=SIZE,outer_lr=LR,probe_norm=PROBE_NORM,steps_per_epoch=STEPS//2)
    with (JOB/f"{arm}_updates.jsonl").open("w") as stream:
        for step,spec in enumerate(schedule,1):
            support,query=episode(ds,spec)
            weights,cross=update(net,opt,names,parameters,support,query,arm)
            stream.write(json.dumps(dict(step=step,rate=spec["rate"],weights=weights,loss_changes=cross))+"\n")
            if step in (STEPS//2,112,STEPS):
                grid=validate(net,dev,labels)
                history.append(dict(step=step, grid=grid))
                score=(min(grid),sum(grid)/len(grid))
                if step>=112 and (best_score is None or score>best_score):
                    best_score=score
                    best_step=step
                    chosen={n:p.detach().cpu().clone() for n,p in zip(names,parameters)}
                print(f"{arm} step {step}: validation worst={score[0]:.3f}, avg={score[1]:.3f}",flush=True)
    metadata.update(best_step=best_step,validation=history)
    ck["cross_meta"]=metadata
    for n,tensor in chosen.items():
        ck["state_dict"]["model."+n]=tensor
    out=JOB/f"{arm}.ckpt"
    torch.save(ck,out.with_suffix(".tmp"))
    out.with_suffix(".tmp").replace(out)
    (JOB/f"{arm}_validation.json").write_text(json.dumps(metadata,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=("prepare","equal","fomaml","learned"))
    arm=parser.parse_args().arm
    if arm=="prepare":
        prepare()
    else:
        train(arm)
