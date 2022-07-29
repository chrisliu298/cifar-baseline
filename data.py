import os
from copy import deepcopy as c

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

DATASETS = {"cifar10": CIFAR10, "cifar100": CIFAR100}


class ImageDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_workers = os.cpu_count()
        # calculate mean and std
        train_dataset = DATASETS[self.config.dataset](
            "/tmp/data", train=True, download=True
        )
        means = (np.mean(train_dataset.data, axis=(0, 1, 2)) / 255.0).round(4).tolist()
        stds = (np.std(train_dataset.data, axis=(0, 1, 2)) / 255.0).round(4).tolist()
        # define transforms
        self.transforms_train = []
        self.transforms_test = []
        base_transfroms = [transforms.ToTensor(), transforms.Normalize(means, stds)]
        if self.config.data_augmentation:
            self.transforms_train.append(transforms.RandomCrop(32, padding=4))
            self.transforms_train.append(transforms.RandomHorizontalFlip())
        self.transforms_train.extend(base_transfroms)
        self.transforms_test.extend(base_transfroms)

    def prepare_data(self):
        # download data
        self.train_dataset = DATASETS[self.config.dataset](
            "/tmp/data",
            train=True,
            download=True,
            transform=transforms.Compose(self.transforms_train),
        )
        self.val_dataset = DATASETS[self.config.dataset](
            "/tmp/data",
            train=True,
            download=True,
            transform=transforms.Compose(self.transforms_test),
        )
        self.test_dataset = DATASETS[self.config.dataset](
            "/tmp/data",
            train=False,
            download=True,
            transform=transforms.Compose(self.transforms_test),
        )

    def setup(self, stage=None):
        self.split_data()

    def split_data(self, val_size=0.2):
        indices = np.arange(len(self.train_dataset))
        if self.config.subset_size:
            indices, _ = train_test_split(
                indices, train_size=self.config.subset_size, shuffle=True
            )
        train_idx, val_idx = train_test_split(indices, test_size=val_size, shuffle=True)
        tmp_train_dataset = c(self.train_dataset)
        tmp_val_dataset = c(self.val_dataset)
        self.train_dataset = Subset(tmp_train_dataset, train_idx)
        self.val_dataset = Subset(tmp_val_dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
