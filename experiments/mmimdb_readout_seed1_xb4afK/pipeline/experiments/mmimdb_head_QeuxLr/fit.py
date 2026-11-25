"""Fit a single MMIMDb head and movie-disjoint validation thresholds."""
import csv
import hashlib
import json
from statistics import mean

import torch
import torch.nn.functional as F

from features import CKPT, JOB, PREV

torch.set_num_threads(3)
torch.manual_seed(1)
WEIGHT = "mmimdb_classifier.weight"
BIAS = "mmimdb_classifier.bias"
SCENARIOS = [(mode, rate) for mode in ("both", "image", "text") for rate in (0.5, 0.7, 0.9)]


def split_movies(ids):
    return torch.tensor([hashlib.sha256(movie.encode()).digest()[0] < 144 for movie in ids])


def fit_thresholds(scores, y):
    # Every plot is seen in all three states. These weights match the mean grid mixture.
    weights = torch.tensor([0.30, 0.35, 0.35])[:, None, None]
    positive = y.bool()[None]
    n_pos = y.sum(0)
    grid = torch.arange(-4., 1.01, .125)
    f1 = []
    for cut in grid:
        pred = scores > cut
        tp = ((pred & positive) * weights).sum((0, 1))
        num_pred = (pred * weights).sum((0, 1))
        f1.append(2 * tp / (num_pred + n_pos).clamp_min(1))
    f1 = torch.stack(f1)
    global_cut = grid[f1.mean(1).argmax()]
    cuts = grid[f1.argmax(0)]
    shrink = n_pos / (n_pos + 30)
    return shrink * cuts + (1 - shrink) * global_cut


def val_grid(scores, labels):
    # Expected confusion counts across independently sampled movie masks.
    positive = labels.bool()[None]
    pred = scores > 0
    tp = (pred & positive).sum(1).float()
    predicted = pred.sum(1).float()
    actual = labels.sum(0)
    metrics = []
    for mode, rate in SCENARIOS:
        text_missing = {"text": rate, "image": 0, "both": rate / 2}[mode]
        weights = torch.tensor([1-rate, text_missing, rate-text_missing])
        metrics.append((2 * (weights @ tp) / ((weights @ predicted) + actual).clamp_min(1)).mean().item()*100)
    return metrics


def refit(train, w0, b0):
    x = train["x"].cuda()
    y = train["y"].float().cuda()
    head = torch.nn.Linear(w0.shape[1], w0.shape[0], device="cuda")
    head.load_state_dict(dict(weight=w0, bias=b0))
    anchor = w0.cuda()
    pos = y.sum(0)
    weights = ((len(y) - pos) / pos.clamp_min(1)).sqrt().clamp(1, 8)
    opt = torch.optim.AdamW(head.parameters(), lr=0.00015, weight_decay=0.01)
    for epoch in range(1, 9):
        for indices in torch.randperm(3*len(y), device="cuda").split(512):
            states, items = indices // len(y), indices % len(y)
            logits = head(x[states, items].float())
            loss = F.binary_cross_entropy_with_logits(logits, y[items], pos_weight=weights)
            loss = loss + 0.005 * (head.weight-anchor).square().sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch in (2, 8):
            yield f"refit_epoch_{epoch}", head.weight.detach().cpu().clone(), head.bias.detach().cpu().clone()


def choose():
    results = []
    for name in ("last", "best"):
        data = torch.load(JOB / f"{name}_dev.pt", map_location="cpu")
        x, y = data["x"].float(), data["y"].float()
        split = split_movies(data["ids"])
        state = torch.load(CKPT[name], map_location="cpu")["state_dict"]
        w, b = state[WEIGHT].float(), state[BIAS].float()
        del state
        candidates = [("original", w, b)]
        if name == "best":
            train = torch.load(JOB / "best_train.pt", map_location="cpu")
            assert not set(train["ids"]).intersection(data["ids"])
            candidates.extend(refit(train, w, b))
            del train
        for method, weight, bias in candidates:
            scores = F.linear(x, weight, bias)
            cuts = fit_thresholds(scores[:,split], y[split])
            for calibrated in (False, True):
                threshold = cuts if calibrated else torch.zeros_like(cuts)
                grid = val_grid(scores[:,~split]-threshold, y[~split])
                results.append((min(grid), mean(grid), name, method, calibrated, grid, weight, bias-threshold))
    # Select on the worst validation condition, then the nine-condition mean.
    worst, avg, name, method, calibrated, grid, w, b = max(results, key=lambda r: r[:2])
    records = [dict(source=n, head=h, calibrated=c, minimum=lo, mean=av, grid=g)
               for lo,av,n,h,c,g,_,_ in results]
    selection = dict(source=name, checkpoint=str(CKPT[name]), head=method, calibrated=calibrated,
                     minimum=worst, mean=avg, scenarios=SCENARIOS, candidates=records)
    (JOB/"selection.json").write_text(json.dumps(selection, indent=2))
    ckpt = torch.load(CKPT[name], map_location="cpu")
    deployment = {k: ckpt[k] for k in ("state_dict", "hyper_parameters", "pytorch-lightning_version")}
    deployment["state_dict"][WEIGHT] = w.clone()
    deployment["state_dict"][BIAS] = b.clone()
    deployment["mmimdb_selection"] = dict(source=name, head=method, calibrated=calibrated)
    torch.save(deployment, JOB/"selected.ckpt")
    print(f"Selected {name}/{method} calibrated={calibrated}, worst={worst:.3f}, mean={avg:.3f}", flush=True)


if __name__ == "__main__":
    choose()
