import csv
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import DataLoader

from clip.datasets.base_dataset import BaseDataset
from clip.modules import CLIPransformerSS, clip
from clip.datamodules.missing_schedule import make_schedule
from pool import Capture, projectors

JOB = Path(os.environ.get("FOOD_JOB", Path(__file__).resolve().parent))
DATA = Path("/disk1/longw/Datasets/food101")
GRID = Path("/disk1/longw/work/mm_sota/missing_grid_20261110/masks/food101")
CASES = [(direction, rate) for direction in ("text", "image", "both") for rate in (50,70,90)]
SEEDS = (42, 2026, 3407)


def setup():
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False


def save(data, path):
    tmp = path.with_suffix(".part")
    torch.save(data, tmp)
    tmp.replace(path)


def csv_out(path, rows):
    with path.with_suffix(".part").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".part").replace(path)


class Food(BaseDataset):
    def __init__(self, split):
        super().__init__(str(DATA), ["CLIP_transform"], 224, ["food101_"+split], text_column_name="text",
                         remove_duplicate=False)
        self.ids = self.table["image_id"].to_pylist()
        self.labels = torch.tensor(self.table["label"].to_pylist())
        assert len(set(self.ids)) == len(self.ids) == len(self) and self.labels.unique().numel() == 101
        assert all(self.index_mapper[i] == (i,0) for i in range(len(self)))
        titles = [v[0] for v in self.all_texts]
        self.tokens = torch.cat([clip.tokenize(titles[j:j+256], context_length=77, truncate=True)
                                 for j in range(0,len(self),256)])

    def __getitem__(self, i):
        return self.get_image(i)["image"][0], self.tokens[i], self.labels[i], i


def loader(ds, batch=64):
    return DataLoader(ds, batch_size=batch, num_workers=5, pin_memory=True)


def warm_checkpoint():
    paths = list((JOB/"checkpoints/food101_curriculum_seed0/version_0/checkpoints").glob("epoch=*.ckpt"))
    assert len(paths) == 1, paths
    return paths[0]


def load_native(path):
    ck = torch.load(path, map_location="cpu")
    cfg = dict(ck["hyper_parameters"]["config"], test_only=True, load_path="")
    assert cfg["loss_names"]["food101"] == 1
    net = CLIPransformerSS(cfg)
    net.load_state_dict(ck["state_dict"], strict=True)
    return net.cuda().eval().requires_grad_(False)


def output(net, img, tok, state, params=None):
    n = len(tok)
    learner = net.model.prompt_learner
    pi, pt = learner([state]) if params is None else functional_call(learner, params, ([state],))
    m = [state]*n
    vi = None if state == 2 else net.model.image_encoder(img.type(net.model.dtype), [p.expand(n,-1,-1) for p in pi], m)
    tx = None if state == 1 else net.model.text_encoder(tok, [p.expand(n,-1,-1) for p in pt], m)
    if vi is None:
        vi = torch.zeros_like(tx)
    if tx is None:
        tx = torch.zeros_like(vi)
    return net.food101_classifier(torch.cat((vi,tx),1)).float(), vi, tx


def local_features(net, cap, state):
    x, valid = cap.local(state)
    if state == 1:
        side = int(x.shape[1] ** .5)
        x = F.adaptive_avg_pool2d(x.float().transpose(1,2).reshape(len(x),-1,side,side), (3,3))
        x = x.flatten(2).transpose(1,2)
        mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    else:
        bins = (torch.arange(x.size(1), device=x.device)[None] * 9 // valid.sum(1,keepdim=True).clamp_min(1)).clamp_max(8)
        member = F.one_hot(bins,9).transpose(1,2).float() * valid[:,None]
        count = member.sum(-1)
        x = (member/count.clamp_min(1)[:,:,None]) @ x.float()
        mask = count > 0
    return (x.half() @ projectors(net)[state-1].half()), mask


def state_features(net, images, tokens, state, *, raw_only=False):
    raw = []
    enc = net.model.image_encoder
    hook = None
    if state == 1:
        hook = enc.ln_post.register_forward_hook(lambda module, args, out: raw.append(out) if out.ndim == 2 else None)
    try:
        capture = nullcontext() if raw_only else Capture(net,{state})
        with torch.autocast("cuda",dtype=torch.float16), capture as cap:
            if raw_only:
                assert state == 1
                pi, _ = net.model.prompt_learner([1])
                n = len(images)
                enc(images.type(net.model.dtype), [p.expand(n,-1,-1) for p in pi], [1]*n)
                assert len(raw) == 1 and raw[0].shape == (n,768)
                return {"raw": raw[0].half()}
            z, vi, tx = output(net,images,tokens,state)
            if state == 0:
                return {"base0": z}
            x, mask = local_features(net,cap,state)
            d = {f"base{state}": z, f"x{state}": x, f"h{state}": vi if state==1 else tx, f"mask{state}": mask}
            if state == 1:
                assert len(raw) == 1 and raw[0].shape == (len(tokens),768)
                d["raw"] = raw[0].half()
            return d
    finally:
        if hook is not None:
            hook.remove()


@torch.inference_mode()
def encode(net, ds, *, features):
    blocks = []
    for image, token, label, _ in loader(ds):
        image, token = image.cuda(), token.cuda()
        d = {"y":label}
        for state in range(3):
            if features:
                v = state_features(net,image,token,state)
            else:
                with torch.autocast("cuda",dtype=torch.float16):
                    v = {f"base{state}":output(net,image,token,state)[0]}
            d.update({k:t.cpu() for k,t in v.items()})
        blocks.append(d)
    return {k:torch.cat([v[k] for v in blocks]) for k in blocks[0]}


def condition_masks(ids, split):
    result = {}
    for kind, rate in CASES:
        for seed in SEEDS:
            if split == "test":
                with (GRID/f"{kind}_{rate}"/f"seed_{seed}.jsonl").open() as f:
                    records = [json.loads(line) for line in f]
                by_id = {r["id"]:int(r["text_missing"])+2*int(r["image_missing"]) for r in records}
                assert len(by_id) == len(records) == len(ids) and set(ids) == by_id.keys()
                state = np.array([by_id[k] for k in ids])
            else:
                order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(f"{seed}/{ids[i]}".encode()).digest())
                chosen = np.array(order[:round(len(ids)*rate/100)])
                state = np.zeros(len(ids),dtype=np.int64)
                nt = len(chosen) if kind == "text" else 0 if kind == "image" else (len(chosen)+seed%2)//2
                state[chosen[:nt]],state[chosen[nt:]] = 1,2
            assert np.all((state>=0)&(state<3))
            result[kind,rate,seed] = torch.as_tensor(state)
    return result


def grid(logits, y, masks):
    pred = logits.argmax(-1).cpu() if logits.ndim==3 else logits.cpu()
    correct = pred == y.cpu()[:,None]
    row = torch.arange(len(y))
    each = {k:100*correct[row,m].double().mean().item() for k,m in masks.items()}
    means = {f"{k}_{r}": float(np.mean([each[k,r,s] for s in SEEDS])) for k,r in CASES}
    return means,each


def mix_masks(n, rate, rng):
    order = rng.permutation(n)
    count = round(n*rate)
    masks = []
    for nt in (count,0,(count+int(rng.integers(2)))//2):
        state = np.zeros(n,dtype=int)
        state[order[:nt]],state[order[nt:count]] = 1,2
        masks.append(state)
    return masks


def extra_indices(n, used, count, rng):
    seen = set(used.tolist())
    out = []
    while len(out)<count:
        i = int(rng.integers(n))
        if i not in seen:
            out.append(i)
            seen.add(i)
    return np.array(out)
