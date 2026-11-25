"""Fit on training posters, lock on development movies, and replay seed 1 from raw test inputs."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

JOB = Path(__file__).resolve().parent
EXP = JOB / "pipeline/experiments"
ARCHIVE = JOB.parent
SEM = EXP / "mmimdb_pair_ablation_dZfki9"
DIAG = ARCHIVE / "mmimdb_projection_diagnosis_Vnn3On"
MATCH = ARCHIVE / "mmimdb_matched_probe_obZdDo"
sys.path[:0] = [str(SEM), str(MATCH), str(DIAG)]
from semantic_io import load, plots, setup, csv_out, store, load_test_masks
from predictor import infer
from compare import ap, logits, thresholds
from probe_fit import fit
from replay_readout import forward, load_readout, paired_intervals, score_counts

SOURCE = SEM / "k1/text90.ckpt"
DEPLOY = JOB / "raw_readout_seed1.ckpt"
MASK_SEEDS = (42, 2026, 3407)


def metadata():
    with (MATCH / "metadata.pkl").open("rb") as f:
        return pickle.load(f)


def verify_data(ds, info):
    rows = np.fromiter((ds.index_mapper[i][0] for i in range(len(ds))), dtype=np.int64)
    assert [str(x).zfill(7) for x in ds.table["image_id"].to_pylist()] == info["ids"]
    np.testing.assert_array_equal(rows, info["rows"])
    np.testing.assert_array_equal(ds.table["label"].to_pylist(), info["labels"])


@torch.inference_mode()
def raw_posters(net, split, info):
    ds = plots(net.hparams.config, split)
    verify_data(ds, info)
    first = np.unique(info["rows"], return_index=True)[1]
    ds = Subset(ds, first.tolist())
    features = []
    encoder = net.model.image_encoder
    captured = []
    hook = encoder.ln_post.register_forward_hook(lambda mod, args, out: captured.append(out))
    try:
        for image, _ in DataLoader(ds, batch_size=64, num_workers=5, pin_memory=True):
            n = len(image)
            with torch.autocast("cuda", dtype=torch.float16):
                z = net.infer(dict(image=[image.cuda()], text=([""] * n,), missing_type=[1] * n))["cls_feats"]
                assert len(captured) == 1
                raw = captured.pop()
                assert raw.shape == (n, 768)
                torch.testing.assert_close(raw @ encoder.proj, z[:, :512], rtol=0, atol=0)
            features.append(raw.half().cpu())
    finally:
        hook.remove()
    return torch.cat(features)


def train():
    setup()
    meta = metadata()
    net, checkpoint = load(SOURCE)
    assert checkpoint["hyper_parameters"]["config"]["seed"] == 1
    assert checkpoint["semantic_predictor"]["semantic_weight"] == 0
    net.cuda().eval().requires_grad_(False)
    tr, dev = meta["train"], meta["dev"]
    assert set(tr["ids"]).isdisjoint(dev["ids"])
    x = raw_posters(net, "train", tr)
    xv = raw_posters(net, "dev", dev).numpy()
    del net
    cal = dev["cal"]
    assert len(x) == 15552 and cal.sum() == 1444
    data = dict(x=x, labels=torch.from_numpy(tr["labels"]), weight=torch.from_numpy(np.bincount(tr["rows"])))
    weights, labels = np.bincount(dev["rows"])[cal], dev["labels"][cal]
    trials, log = [], []
    for ridge in (.0003, .003, .03, .3, 3., 30., 300.):
        w, b, grad = fit(data, torch.zeros(23, 768), torch.zeros(23), ridge)
        z = logits(xv[cal], w, b)
        score = ap(labels, z, weights)
        trials.append((score, ridge, w, b, z))
        log.append(dict(ridge=ridge, cal_macro_AP=score, max_gradient=grad))
    _, ridge, w, b, z = max(trials, key=lambda t: (t[0], t[1]))
    checkpoint["poster_readout"] = dict(weight=w, bias=b, thresholds=torch.as_tensor(thresholds(labels, z, weights)),
                                         ridge=ridge, input="image_encoder.ln_post")
    csv_out(JOB / "readout_calibration.csv", log)
    store(DEPLOY, checkpoint)
    print("Locked", SOURCE.resolve(), ridge, len(x), int(cal.sum()), flush=True)


@torch.inference_mode()
def test():
    setup()
    net, gen, head = load_readout(DEPLOY)
    info = metadata()["test"]
    ds = plots(net.hparams.config, "test")
    verify_data(ds, info)
    rows = info["rows"]
    y = info["labels"][rows]
    with (DIAG / "per_mask.csv").open() as stream:
        ref = {(r["mode"], int(r["rate"]), int(r["seed"])): r for r in csv.DictReader(stream)}
    masks = load_test_masks(info["ids"], ("text", "both", "image"), (50, 70, 90))
    candidate, replay = [], []
    for step, (image, texts) in enumerate(DataLoader(ds, batch_size=64, num_workers=5, pin_memory=True)):
        image = image.cuda()
        batch_pred, batch_base = [], []
        for state in (0, 1, 2):
            batch = dict(image=[torch.zeros_like(image) if state == 2 else image],
                         text=([""] * len(image) if state == 1 else texts,), missing_type=[state] * len(image))
            pred = forward(net, gen, head, batch).cpu().numpy() > 0
            batch_pred.append(pred)
            batch_base.append(infer(net, gen, batch).cpu().numpy() > 0 if state == 1 else pred)
        if step == 0:
            modes = np.arange(len(image)) % 3
            batch = dict(image=[image.masked_fill(torch.as_tensor(modes == 2, device="cuda")[:, None, None, None], 0)],
                         text=(["" if s == 1 else t for s, t in zip(modes, texts)],), missing_type=modes.tolist())
            mixed = forward(net, gen, head, batch).cpu().numpy() > 0
            np.testing.assert_array_equal(mixed, np.stack(batch_pred, 1)[np.arange(len(image)), modes])
        candidate.append(np.stack(batch_base, 1))
        replay.append(np.stack(batch_pred, 1))
    deployed, base = np.concatenate(replay), np.concatenate(candidate)
    assert len(deployed) == len(rows) == 16120
    idx = np.arange(len(rows))
    per_mask, grid = [], []
    high = {k: [] for k in ("replay", "main", "dpl")}
    with np.load(MATCH / "dpl.npz") as f:
        poster, complete = f["test_pred"][rows], f["complete_pred"]
    for mode in ("text", "both", "image"):
        for rate in (50, 70, 90):
            sample = []
            for seed in MASK_SEEDS:
                states = masks[mode, rate, seed].numpy()[rows]
                p = deployed[idx, states]
                original = base[idx, states]
                compare = ref[mode, rate, seed]
                vals = dict(seed1=float(score_counts(y, p)), seed1_without_readout=float(score_counts(y, original)),
                            seed0=float(compare["raw_probe"]), dpl=float(compare["dpl_expert"]))
                vals["seed1_minus_dpl"] = vals["seed1"] - vals["dpl"]
                vals["readout_gain"] = vals["seed1"] - vals["seed1_without_readout"]
                sample.append(vals)
                per_mask.append(dict(mode=mode, rate=rate, mask_seed=seed, **vals))
                if (mode, rate) == ("text", 90):
                    cp = info["complete_indices"]
                    assert np.all(np.isin(idx[states == 0], cp))
                    which = states[cp] == 0
                    specialist = poster.copy()
                    specialist[cp[which]] = complete[which]
                    assert abs(score_counts(y, specialist) - vals["dpl"]) < .001
                    for k, v in (("replay", p), ("main", original), ("dpl", specialist)):
                        high[k].append(v)
            grid.append(dict(mode=mode, rate=rate, **{k: float(np.mean([r[k] for r in sample])) for k in sample[0]}))
    csv_out(JOB / "per_mask.csv", per_mask)
    csv_out(JOB / "test_grid.csv", grid)
    csv_out(JOB / "paired_text90.csv", paired_intervals(y, rows, {k: np.stack(v, 1) for k, v in high.items()}))
    summary = dict(train_seed=1, test_movies=len(info["ids"]), test_plots=len(rows), test_masks=MASK_SEEDS,
                   mean9={k: float(np.mean([r[k] for r in grid])) for k in sample[0]},
                   wins=sum(r["seed1_minus_dpl"] > 0 for r in grid), text50=grid[0], text90=grid[2])
    (JOB / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    {"fit": train, "test": test}[sys.argv[1]]()
