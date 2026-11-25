import argparse
import json

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from common import JOB, INITIAL, TUNED, load, plots, loader, val_grid, split_movies


class Posters(Dataset):
    def __init__(self, cfg, split, data):
        self.source = plots(cfg, split)
        assert self.source.table["image_id"].to_pylist() == data["ids"]
        self.first_plot = {}
        for index, (row, _) in self.source.index_mapper.items():
            self.first_plot.setdefault(row, index)
        assert len(self.first_plot) == len(data["ids"])

    def __len__(self):
        return len(self.first_plot)

    def __getitem__(self, index):
        return self.source.get_image(self.first_plot[index])["image"][0], index


def predict_posters(net, images):
    model = net.model
    learner = model.prompt_learner
    shared = learner.common_prompt_projection_image(learner.common_prompt_image)
    prompts = [shared.expand(len(images), -1, -1)] * learner.prompt_depth
    image_features = model.image_encoder(images.type(model.dtype), prompts, [1]*len(images))
    return net.mmimdb_classifier(torch.cat([image_features, torch.zeros_like(image_features)], dim=1))


@torch.inference_mode()
def validate(net, iterator, data):
    values = []
    with torch.autocast("cuda", dtype=torch.float16):
        for images, _ in iterator:
            values.append(predict_posters(net, images.cuda()).float().cpu())
    scores = data["logits"].clone()
    rows = data["rows"]
    scores[1] = torch.cat(values)[rows]
    selected = ~split_movies([str(data["ids"][i]).zfill(7) for i in rows.tolist()])
    return val_grid(scores[:, selected], data["labels"][rows][selected])


def train(arm):
    torch.manual_seed(0)
    net, ck = load(INITIAL)
    net.requires_grad_(False).cuda().eval()
    params = dict(net.named_parameters())
    for name in TUNED:
        params[name].requires_grad_(True)
    assert {n for n,p in params.items() if p.requires_grad} == set(TUNED)
    cfg = ck["hyper_parameters"]["config"]
    data = torch.load(JOB/"train.pt")
    dev = torch.load(JOB/"dev.pt")
    assert set(data["ids"]).isdisjoint(dev["ids"])
    train_loader = loader(Posters(cfg, "train", data), shuffle=True)
    dev_loader = loader(Posters(cfg, "dev", dev))
    y, teacher = data["labels"].cuda(), data["teacher"].cuda()
    positives = y.sum(0).clamp_min(1)
    pos_weight = ((len(y)-positives)/positives).sqrt().clamp(1, 8)
    opt = torch.optim.AdamW([params[n] for n in TUNED], lr=0.0005, weight_decay=0)
    scaler = torch.cuda.amp.GradScaler()
    best = -float("inf")
    history = []
    for epoch in range(1, 3):
        for step, (images, rows) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = predict_posters(net, images)
                if epoch == 1 and step == 0:
                    with torch.no_grad():
                        size = min(4,len(images))
                        batch = {"image":[images[:size]],"text":([""]*size,),
                                 "missing_type":[1]*size}
                        native = net.mmimdb_classifier(net.infer(batch)["cls_feats"])
                        torch.testing.assert_close(logits[:size],native,atol=.02,rtol=.01)
                loss = F.binary_cross_entropy_with_logits(logits.float(), y[rows], pos_weight=pos_weight)
                if arm == "distilled":
                    loss = loss + 0.25 * 2**2 * F.binary_cross_entropy_with_logits(
                        logits.float()/2, teacher[rows])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if epoch == 1 and step == 0:
                for name in TUNED:
                    grad = params[name].grad
                    assert grad is not None and grad.isfinite().all() and grad.abs().max() > 0, name
            torch.nn.utils.clip_grad_norm_([params[n] for n in TUNED], 1.)
            scaler.step(opt)
            scaler.update()
        scores = validate(net,dev_loader,dev)
        history.append(dict(epoch=epoch,grid=scores))
        if scores[-1] > best:
            best = scores[-1]
            best_epoch = epoch
            update = {n:params[n].detach().cpu().clone() for n in TUNED}
        print(f"{arm}: epoch {epoch}, validation missing-text 90% Macro-F1 {scores[-1]:.3f}",flush=True)

    original = ck["state_dict"]
    for n in TUNED:
        assert not torch.equal(update[n], original[n])
        original[n] = update[n]
    ck["visual_prompt_tuning"] = dict(arm=arm, epoch=best_epoch, seed=0,
                                      trainable_names=list(TUNED), grid=history)
    path = JOB/f"{arm}.ckpt"
    torch.save(ck,path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    (JOB/f"{arm}_validation.json").write_text(json.dumps(ck["visual_prompt_tuning"],indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["supervised","distilled"])
    train(parser.parse_args().arm)
