"""Replay the warm, last and selected meta checkpoints on the same validation memes."""
import json
from pathlib import Path

import numpy as np
import torch

from food_io import JOB, SingleLabelDataset, condition_masks, csv_out, encode, grid, load_native, setup
from single_label import measure

OUT = Path(__file__).resolve().parent
SOURCES = {
    0: JOB,
    2: JOB.parent / "hatememes_rank_seeds_BkwGOl/seed2",
    1: JOB.parent / "hatememes_rank_seeds_BkwGOl/seed1",
    42: JOB.parent / "hatememes_rank_seeds42_3047_L60uJ1/seed42",
    3047: JOB.parent / "hatememes_rank_seeds42_3047_L60uJ1/seed3047",
}


def main():
    setup()
    dataset = SingleLabelDataset("val")
    masks = condition_masks(dataset.ids, "val")
    rows = []
    for seed, job in SOURCES.items():
        assert json.loads((job / "verified_ids.json").read_text())["val"] == dataset.ids
        checkpoint_dir = job / f"checkpoints/hatememes_curriculum_seed{seed}/version_0/checkpoints"
        warm, = checkpoint_dir.glob("epoch=*.ckpt")
        archive = json.loads((job / "prompt_selection.json").read_text())
        for stage, path in (("warm", warm), ("last", checkpoint_dir / "last.ckpt"),
                            ("meta", job / "prompt_selected.ckpt")):
            net = load_native(path)
            assert net.hparams.config["seed"] == seed
            data = encode(net, dataset, features=False)
            del net
            assert torch.equal(data["y"], dataset.labels)
            logits = torch.stack([data[f"base{s}"] for s in range(3)], 1)
            margins = logits[:,:,1].double() - logits[:,:,0].double()
            metrics, _ = grid(margins, dataset.labels, masks)
            if stage == "warm":
                assert max(abs(metrics[k] - archive["baseline"][k]) for k in metrics) < .02
            if stage == "meta":
                epoch = archive["selected"]
                expected = archive["history"][epoch-1]["auroc"] if epoch else archive["baseline"]
                assert max(abs(metrics[k] - expected[k]) for k in metrics) < .02
            else:
                epoch = int(torch.load(path, map_location="cpu")["epoch"])
            pure = {k: measure(margins[:,s], dataset.labels) for s,k in enumerate(("complete","only_image","only_text"))}
            rows.append(dict(seed=seed,stage=stage,epoch=epoch,mean9=np.mean(list(metrics.values())),**metrics,**pure))
            print(seed, stage, epoch, metrics["image_90"], flush=True)
        csv_out(OUT / "curriculum_validation.csv", rows)


if __name__ == "__main__":
    main()
