"""Evaluate fixed checkpoints on the archived MMIMDb per-ID masks."""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_f1_score

GRID = Path("/disk1/longw/work/mm_sota/missing_grid_20261110")
DATA = Path("/disk1/longw/Datasets/mmimdb")
SEEDS = (42, 2026, 3407)
RATES = (50, 70, 90)
MODES = ("both", "image", "text")


def load_test_masks(ids, modes, rates):
    unique_ids = set(ids)
    masks = {}
    for mode in modes:
        for rate in rates:
            for seed in SEEDS:
                path = GRID / "masks" / "mmimdb" / f"{mode}_{rate}" / f"seed_{seed}.jsonl"
                by_id = {}
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    sample_id = row["id"]
                    text_missing = row["text_missing"]
                    image_missing = row["image_missing"]
                    if sample_id in by_id or text_missing and image_missing:
                        raise ValueError(f"Invalid mask entry for {sample_id} in {path}")
                    by_id[sample_id] = int(text_missing) + 2 * int(image_missing)
                if by_id.keys() != unique_ids:
                    raise ValueError(f"Test IDs do not match {path}")
                masks[mode, rate, seed] = torch.tensor([by_id[item_id] for item_id in ids])
    return masks


def evaluate(args):
    sys.path.insert(0, str(args.repo.resolve()))
    from clip.datasets.base_dataset import BaseDataset
    from clip.modules import CLIPransformerSS

    class TestDataset(BaseDataset):
        def __getitem__(self, index):
            return {
                "image": self.get_image(index)["image"][0],
                "text": self.get_text(index)["text"],
            }

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = dict(checkpoint["hyper_parameters"]["config"])
    config.update(test_only=True, load_path="")
    model = CLIPransformerSS(config).eval()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    device = torch.device("cuda")
    model.to(device)

    test_data = TestDataset(
        data_dir=str(DATA),
        transform_keys=config["val_transform_keys"],
        image_size=config["image_size"],
        names=["mmimdb_test"],
        text_column_name="plots",
        remove_duplicate=False,
    )
    image_ids = test_data.table["image_id"].to_pylist()
    image_indices = [test_data.index_mapper[index][0] for index in range(len(test_data))]
    ids = [str(image_ids[index]).zfill(7) for index in image_indices]
    labels = torch.tensor(test_data.table["label"].to_pylist()).long()[image_indices]
    if len(ids) != 16120 or len(set(ids)) != 7799:
        raise ValueError("Expected 16120 plot items from 7799 MMIMDb test movies.")
    masks = load_test_masks(ids, MODES, RATES)

    loader = DataLoader(test_data, batch_size=64, num_workers=8, pin_memory=True)
    logits_by_type = [[], [], []]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            text = batch["text"]
            size = len(text)
            for missing_type, logits in enumerate(logits_by_type):
                masked = {
                    "image": [torch.zeros_like(image) if missing_type == 2 else image],
                    "text": ([""] * size if missing_type == 1 else text,),
                    "missing_type": [missing_type] * size,
                }
                output = model.mmimdb_classifier(model.infer(masked)["cls_feats"])
                logits.append(output.float().cpu())

    logits = torch.stack([torch.cat(parts) for parts in logits_by_type])
    indices = torch.arange(len(test_data))
    rows = []
    for (mode, rate, seed), mask in masks.items():
        score = multilabel_f1_score(
            logits[mask, indices].sigmoid(), labels, num_labels=23, average="macro"
        ).item()
        rows.append(
            {
                "method": args.method,
                "missing_type": mode,
                "missing_rate": rate / 100,
                "mask_seed": seed,
                "macro_f1_percent": 100 * score,
                "checkpoint": str(args.checkpoint.resolve()),
            }
        )
    temporary = args.output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"{args.method}: {len(rows)} matched mask scores in {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", type=Path, required=True)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
