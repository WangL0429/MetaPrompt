import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

JOB = Path(__file__).resolve().parent
ROOT = JOB.parents[1]
PREV = ROOT / "experiments/mmimdb_curriculum_20261119"
WEIGHTS = PREV / "checkpoints/curriculum_seed1/version_0/checkpoints"
CKPT = {"best": next(WEIGHTS.glob("epoch=*.ckpt")), "last": WEIGHTS / "last.ckpt"}
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PREV))
from evaluate_mmimdb import DATA
from clip.datasets.base_dataset import BaseDataset
from clip.modules import CLIPransformerSS


class PlotDataset(BaseDataset):
    def __getitem__(self, index):
        return self.get_image(index)["image"][0], self.get_text(index)["text"]


def extract(name, split):
    checkpoint = torch.load(CKPT[name], map_location="cpu")
    config = dict(checkpoint["hyper_parameters"]["config"], test_only=True, load_path="")
    torch.manual_seed(1)
    model = CLIPransformerSS(config).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.cuda()
    dataset = PlotDataset(data_dir=str(DATA), transform_keys=config["val_transform_keys"],
                          image_size=config["image_size"], names=[f"mmimdb_{split}"],
                          text_column_name="plots", remove_duplicate=False)
    movie_ids = dataset.table["image_id"].to_pylist()
    rows = [dataset.index_mapper[index][0] for index in range(len(dataset))]
    ids = [str(movie_ids[row]).zfill(7) for row in rows]
    labels = torch.tensor(dataset.table["label"].to_pylist(), dtype=torch.uint8)[rows]
    cached = [[], [], []]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for images, texts in DataLoader(dataset, batch_size=64, num_workers=6, pin_memory=True):
            images = images.cuda(non_blocking=True)
            for state, chunks in enumerate(cached):
                batch = {"image": [torch.zeros_like(images) if state == 2 else images],
                         "text": ([""] * len(texts) if state == 1 else texts,),
                         "missing_type": [state] * len(texts)}
                chunks.append(model.infer(batch)["cls_feats"].cpu())
    path = JOB / f"{name}_{split}.pt"
    torch.save(dict(x=torch.stack([torch.cat(chunks) for chunks in cached]),
                    y=labels, ids=ids), path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    print(f"Extracted {split}: {len(rows)} plots, {len(set(ids))} movies.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=CKPT)
    parser.add_argument("split", choices=["dev", "train", "test"])
    args = parser.parse_args()
    extract(args.name, args.split)
