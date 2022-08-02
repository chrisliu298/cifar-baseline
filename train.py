import argparse
import json
import logging
import os
import warnings

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
    # data
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=list(DATASETS.keys())
    )
    parser.add_argument("--data_augmentation", action="store_true")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--label_noise_type", type=str, choices=["a", "s"])
    parser.add_argument("--label_noise_level", type=float, default=0.0)
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
    # assign additional args
    config.num_classes = 10 if config.dataset == "cifar10" else 100
    config.lr = (
        float(
            loguniform.rvs(1e-4, 1e-2)
            if "adam" in config.optimizer
            else loguniform.rvs(1e-3, 1e-1)
        )
        or config.lr
    )
    config.wd = (
        float(
            loguniform.rvs(1e-4, 1)
            if "adam" in config.optimizer
            else loguniform.rvs(1e-4, 1e-3)
        )
        or config.wd
    )
    # set seed for reproducibility
    seed_everything(config.seed)
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
    trainer = Trainer(
        gpus=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=WandbLogger(
            offline=not config.wandb,
            project=config.project_id,
            entity="chrisliu298",
            config=config,
        ),
        profiler="simple",
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, verbose=config.verbose)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
