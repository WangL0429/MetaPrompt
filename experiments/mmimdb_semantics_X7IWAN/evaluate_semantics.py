"""Evaluate only the selected checkpoints; round-trip through the raw-input inference path."""
import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from predictor import load_predictor_checkpoint, infer
from fit_semantics import read, predictions
from semantic_io import JOB, setup, metadata, plots, val_grid, store, json_out


@torch.inference_mode()
def check_native(net, generator, sample, cached):
    image, texts = sample
    image = image.cuda()
    size = len(image)
    checks = {}
    for name in ("complete", "text", "image", "mixed"):
        states = torch.arange(size, device="cuda") % 3 if name == "mixed" else torch.full(
            (size,), ("complete", "text", "image").index(name), device="cuda")
        modes = states.tolist()
        masked_image = image.masked_fill((states == 2)[:, None, None, None], 0)
        masked_text = ["" if mode == 1 else t for mode, t in zip(modes, texts)]
        batch = dict(image=[masked_image], text=(masked_text,), missing_type=modes)
        raw = infer(net, generator, batch).cpu()
        ref = cached[torch.arange(size), states.cpu()]
        checks[name] = float((raw-ref).abs().max())
        torch.testing.assert_close(raw, ref, atol=0.015 if name == "mixed" else 0.003, rtol=.003)
    return checks


def evaluate(k):
    setup()
    out = JOB / f"k{k}"
    selection = json.loads((out/"selection.json").read_text())
    dev, test = read("dev"), read("test")
    cal = dev["cal"].cpu()
    labels = dev["y"].cpu()
    info = metadata()["test"]
    assert len(info["ids"]) == test["rows"].unique().numel() == 7799
    assert np.array_equal(test["rows"].cpu().numpy(), info["rows"])
    stats = {}
    for epoch in sorted(set(selection["picks"].values())):
        net, generator = load_predictor_checkpoint(out/f"deploy_epoch{epoch}.ckpt")
        head = net.mmimdb_classifier.weight.detach().float()
        w = {1: head[:,512:], 2: head[:,:512]}
        dev_scores, _ = predictions(generator, dev, w)
        test_scores, diagnostics = predictions(generator, test, w)
        assert torch.equal(test_scores[:,0], test["base"][:,0].cpu())
        assert torch.isfinite(test_scores).all()
        calibration = val_grid(dev_scores[cal].transpose(0,1), labels[cal])
        assert np.allclose(calibration, selection["history"][epoch-1]["cal_grid"], rtol=0, atol=1e-5)
        cfg = net.hparams.config
        sample = next(iter(DataLoader(plots(cfg,"test"), batch_size=64, num_workers=0)))
        native = check_native(net, generator, sample, test_scores)
        stats[str(epoch)] = dict(native_max_abs_logit_error=native, gate_entropy_separation_abslogit=diagnostics,
            heldout_grid=val_grid(dev_scores[~cal].transpose(0,1),labels[~cal]))
        store(out/f"test_epoch_{epoch}.pt", test_scores)
        del net, generator
    json_out(out/"test_diagnostics.json", stats)
    print(k,selection["picks"],stats,flush=True)


if __name__ == "__main__":
    evaluate(int(sys.argv[1]))
