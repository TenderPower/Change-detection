import numpy as np
import pytorch_lightning as pl
from loguru import logger as L
from torch.utils.data import ConcatDataset, DataLoader
import time
from data.inpainted_coco_dataset import InpatinedCocoDataset  # noqa
from data.kubric_change import KubricChange  # noqa
from data.std import StdDataset  # noqa
from data.synthtext_dataset import SynthTextDataset  # noqa
from data.kc3d import KC3D
from data.rc3d import RC3D
from models.test___ import Test
from data.customdatasets import Datasets
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.num_dataloader_workers = args.num_dataloader_workers
        self.method = args.method
        self.dataset_configs = args.datasets
        self.dataloader_collate_fn = self.import_method_specific_functions(self.method)
        self.testAlignImage = Test()
        conf = get_config("zoedepth_nk", "infer")
        depth_predictor = build_model(conf)
        self.depth_predictor = depth_predictor

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("InpaintedCOCODataModule")
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--num_dataloader_workers", type=int)
        parser.add_argument("--test_batch_size", type=int)
        return parent_parser

    def import_method_specific_functions(self, method):
        if method == "centernet" or method == "other":
            from models.centernet_with_coam import dataloader_collate_fn
        else:
            raise NotImplementedError(f"Unknown method {method}")
        return dataloader_collate_fn

    def collate_fn(self, batch, dataset):
        """
        A wrapper collate function that calls method-specific,
        data collation functions. It also takes care of filtering out
        any None batch items and if the batch ends up empty, it attempts
        to create a batch of a single non-None item.
        """
        batch = [x for x in batch if x is not None]
        tries = 0
        while len(batch) == 0:
            tries += 1
            random_item = dataset[np.random.randint(0, len(dataset))]
            if random_item is not None:
                batch = [random_item]
            if tries % 50 == 0:
                L.log(
                    "DEBUG",
                    f"Made {tries} attempts to construct a non-None batch.\
                        If this happens too often, maybe it's not a good workaround.",
                )
        return self.dataloader_collate_fn(batch, self.testAlignImage, self.depth_predictor)

    def setup(self, stage=None):
        train_dataset_configs = self.dataset_configs["train_datasets"]
        train_datasets = []
        for train_dataset_config in train_dataset_configs:
            train_datasets.append(
                eval(train_dataset_config["class"])(self.depth_predictor, **train_dataset_config["args"]))
        self.traindatasets = Datasets(train_datasets)

        val_dataset_configs = self.dataset_configs["val_datasets"]
        val_datasets = []
        for val_dataset_config in val_dataset_configs:
            val_datasets.append(eval(val_dataset_config["class"])(self.depth_predictor, **val_dataset_config["args"]))
        self.valdatasets = Datasets(val_datasets)
        test_datasets_configs = self.dataset_configs["test_datasets"]
        self.test_dataset_names = []
        self.test_datasets = []
        for test_dataset_config in test_datasets_configs:
            if test_dataset_config["class"] == "ConcatDataset":
                datasets = []
                for dataset_config in test_dataset_config["datasets"]:
                    datasets.append(eval(dataset_config["class"])(self.depth_predictor, **dataset_config["args"]))
                self.test_datasets.append(ConcatDataset(datasets))
            else:
                self.test_datasets.append(
                    eval(test_dataset_config["class"])(self.depth_predictor, **test_dataset_config["args"])
                )
            self.test_dataset_names.append(test_dataset_config["name"])

    def train_dataloader(self):
        def collate_fn_wrapper(batch):
            return self.collate_fn(batch, self.traindatasets)

        return DataLoader(self.traindatasets,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_dataloader_workers,
                          collate_fn=collate_fn_wrapper)

    def val_dataloader(self):
        def collate_fn_wrapper(batch):
            return self.collate_fn(batch, self.valdatasets)

        return DataLoader(self.valdatasets,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_dataloader_workers,
                          collate_fn=collate_fn_wrapper)

    def test_dataloader(self):
        dataloaders = []
        for test_dataset in self.test_datasets:
            def collate_fn_wrapper(batch):
                return self.collate_fn(batch, test_dataset)

            dataloaders.append(
                DataLoader(
                    test_dataset,
                    batch_size=self.test_batch_size,
                    num_workers=self.num_dataloader_workers,
                    collate_fn=collate_fn_wrapper,
                )
            )
        return dataloaders
