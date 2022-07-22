import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

DATASETS = {"cifar10": CIFAR10, "cifar100": CIFAR100}


class ImageDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_workers = int(os.cpu_count() / 2)
        # define transforms
        self.train_transforms = []
        self.test_transforms = []
        base_transfroms = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        if self.config.data_augmentation:
            self.train_transforms.append(transforms.RandomCrop(32, padding=4))
            self.train_transforms.append(transforms.RandomHorizontalFlip())
        self.train_transforms.extend(base_transfroms)
        self.test_transforms.extend(base_transfroms)

        # download data
        self.train_dataset = DATASETS[self.config.dataset](
            "/tmp/data",
            train=True,
            download=True,
            transform=transforms.Compose(self.train_transforms),
        )
        self.test_dataset = DATASETS[self.config.dataset](
            "/tmp/data",
            train=False,
            download=True,
            transform=transforms.Compose(self.test_transforms),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
