"""Deterministic per-batch missing modality schedules."""

import random
from itertools import islice

from torch.utils.data.distributed import DistributedSampler


def make_schedule(num_batches, seed, order):
    if order not in {"curriculum", "random"}:
        raise ValueError(f"Unknown missing schedule: {order}")
    rng = random.Random(seed)
    batches = [
        (rng.betavariate(3, 2), rng.choice(("text", "image", "both")), rng.getrandbits(64))
        for _ in range(num_batches)
    ]
    if order == "curriculum":
        cutoff = num_batches - num_batches // 5
        # The original order inside each difficulty band and the mixed suffix are random.
        batches[:cutoff] = sorted(
            batches[:cutoff], key=lambda batch: (batch[0] >= 0.7, batch[0] >= 0.4)
        )
    return batches


class MissingScheduleSampler(DistributedSampler):
    """Pass one planned missing state per sample to the dataset workers."""

    def __init__(self, dataset, batch_size, epochs, order, seed, **kwargs):
        super().__init__(dataset, shuffle=True, seed=seed, **kwargs)
        self.batch_size = batch_size
        self.batches_per_epoch = self.num_samples // batch_size
        if self.batches_per_epoch == 0:
            raise ValueError("A missing schedule requires at least one full batch per rank.")
        self.schedule = make_schedule(epochs * self.batches_per_epoch, seed, order)

    def __len__(self):
        return self.batches_per_epoch * self.batch_size

    def __iter__(self):
        indices = super().__iter__()
        start = self.epoch * self.batches_per_epoch
        for ratio, mode, seed in self.schedule[start : start + self.batches_per_epoch]:
            missing_count = round(self.batch_size * ratio)
            if mode == "both":
                image_count = missing_count // 2
                text_count = missing_count // 2
                if missing_count % 2:
                    if seed % 2:
                        image_count += 1
                    else:
                        text_count += 1
            else:
                text_count = missing_count if mode == "text" else 0
                image_count = missing_count if mode == "image" else 0

            mask = [0] * (self.batch_size - missing_count)
            mask.extend([1] * text_count)
            mask.extend([2] * image_count)
            random.Random(seed + self.rank).shuffle(mask)
            for index, missing_type in zip(islice(indices, self.batch_size), mask):
                yield index, missing_type, ratio
