import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules
from .missing_schedule import MissingScheduleSampler


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist
        self.missing_schedule = None if _config["test_only"] else _config.get("train_missing_schedule")
        self.epochs = _config["max_epoch"]
        self.seed = _config["seed"]
        if self.missing_schedule:
            if len(self.dms) != 1 or _config["max_steps"] is not None:
                raise ValueError("A missing schedule requires one dataset and an epoch-based run.")
            if any(_config.get(f"{task}_meta") for task in ("mmimdb", "food101", "hatememes")):
                raise ValueError("A batch-level missing schedule requires ordinary classification training.")

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        for dm in self.dms:
            dm.setup(stage)

        if self.missing_schedule:
            self.train_dataset = self.dms[0].train_dataset
        else:
            self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])

        self.collate = functools.partial(
            self.dms[0].train_dataset.collate
        )

        if self.missing_schedule:
            distribution = {} if self.dist else {"num_replicas": 1, "rank": 0}
            self.train_sampler = MissingScheduleSampler(
                self.train_dataset,
                self.batch_size,
                self.epochs,
                self.missing_schedule,
                self.seed,
                **distribution,
            )
        else:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True) if self.dist else None

        if self.dist:
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
