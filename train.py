import argparse
import json
import logging
import os
import random
import warnings

import torch
import wandb
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from scipy.stats import loguniform

from data import DATASETS, ImageDataModule
from model import MODELS, Model


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, default="baseline-cifar10")
    # model
    parser.add_argument(
        "--model", type=str, default="resnet18", choices=list(MODELS.keys())
    )
    parser.add_argument("--width_factor", type=int, default=64)
    # data
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=list(DATASETS.keys())
    )
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--noise_type", type=str, choices=["a", "s"])
    parser.add_argument("--noise_rate", type=float, default=0.0)
    # training
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--wd", type=float)
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    config = EasyDict(vars(parser.parse_args()))
    # set seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # conditional args
    config.num_classes = 10 if config.dataset == "cifar10" else 100
    if config.lr is None:
        config.lr = float(
            loguniform.rvs(1e-4, 1e-2)
            if "ada" in config.optimizer
            else loguniform.rvs(1e-2, 0.3)
        )
    if config.wd is None:
        config.wd = float(loguniform.rvs(1e-4, 10))
    # show nothing in stdout
    if not config.verbose:
        os.environ["WANDB_SILENT"] = "True"
        warnings.filterwarnings("ignore")
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    else:
        print(json.dumps(config, indent=4, sort_keys=True))
    # setup data module, model, and trainer
    datamodule = ImageDataModule(config)
    model = Model(config)
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            filename="{epoch}_{avg_val_acc}",
            monitor="avg_val_acc",
            save_top_k=5,
            mode="max",
        )
    )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if not config.verbose:
        callbacks.append(TQDMProgressBar(refresh_rate=0))
    logger = WandbLogger(
        offline=not config.wandb,
        project=config.project_id,
        entity="chrisliu298",
        config=config,
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
        profiler="simple",
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
