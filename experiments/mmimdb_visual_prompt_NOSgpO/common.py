import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[1]
HEAD = JOB.parent / "mmimdb_head_QeuxLr"
PREVIOUS = JOB.parent / "mmimdb_curriculum_20261119"
sys.path.insert(0, str(HEAD))
from features import DATA, PlotDataset, CLIPransformerSS
from fit import val_grid, split_movies

INITIAL = JOB / "initial.ckpt"
COMMON = "model.prompt_learner.common_prompt_image"
DYNAMIC = "model.image_encoder.transformer.resblocks.0.prompts_dynamic_image"
TUNED = (COMMON, DYNAMIC)


def load(path):
    torch.manual_seed(0)
    ck = torch.load(path, map_location="cpu")
    cfg = ck["hyper_parameters"]["config"]
    net = CLIPransformerSS(dict(cfg, test_only=True, load_path="")).eval()
    net.load_state_dict(ck["state_dict"], strict=True)
    return net, ck


def plots(cfg, split):
    return PlotDataset(data_dir=str(DATA), transform_keys=cfg["val_transform_keys"],
                       image_size=cfg["image_size"], names=[f"mmimdb_{split}"],
                       text_column_name="plots", remove_duplicate=False)


def loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=48, num_workers=6, pin_memory=True, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(0))


def initialize():
    net, ck = load(HEAD / "selected.ckpt")
    assert all(n in dict(net.named_parameters()) for n in TUNED)
    for key, value in ck["state_dict"].items():
        assert torch.equal(net.state_dict()[key], value), key
    ck["state_dict"] = net.state_dict()
    torch.save(ck, INITIAL)


@torch.inference_mode()
def cache(split):
    net, ck = load(INITIAL)
    ds = plots(ck["hyper_parameters"]["config"], split)
    del ck
    net.cuda()
    ids = ds.table["image_id"].to_pylist()
    rows = torch.tensor([ds.index_mapper[i][0] for i in range(len(ds))])
    labels = torch.tensor(ds.table["label"].to_pylist()).float()
    states = (0,) if split == "train" else (0, 1, 2)
    chunks = {s: [] for s in states}
    with torch.autocast("cuda", dtype=torch.float16):
        for image, text in loader(ds):
            image = image.cuda(non_blocking=True)
            for s in states:
                batch = {"image": [torch.zeros_like(image) if s==2 else image],
                         "text": ([""]*len(text) if s==1 else text,),
                         "missing_type": [s]*len(text)}
                chunks[s].append(net.mmimdb_classifier(net.infer(batch)["cls_feats"]).float().cpu())
    logits = torch.stack([torch.cat(chunks[s]) for s in states])
    if split == "train":
        soft = (logits[0]/2).sigmoid()
        target = torch.zeros_like(labels).index_add_(0, rows, soft)
        target /= torch.bincount(rows, minlength=len(labels))[:,None]
        assert target.isfinite().all()
        artifact = dict(ids=ids, labels=labels, teacher=target)
    else:
        artifact = dict(ids=ids, labels=labels, rows=rows, logits=logits)
    torch.save(artifact, JOB/f"{split}.pt")


if __name__ == "__main__":
    if sys.argv[1] == "init":
        initialize()
    else:
        cache(sys.argv[1])
