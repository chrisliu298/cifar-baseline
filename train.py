import argparse
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

from data import ImageDataModule, DATASETS
from model import Model, MODELS

# os.environ["WANDB_SILENT"] = "True"
# warnings.filterwarnings("ignore")
# logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", type=str, default="cifar")
    # data
    parser.add_argument("--data_augmentation", action="store_true")
    # training
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"]
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1.0)
    # experiment
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    config = EasyDict(vars(parser.parse_args()))
    seed_everything(config.seed)  # set seed for reproducibility
    # assign additional args
    config.dataset = config.project_id.split("_")[0]
    config.model = config.project_id.split("_")[1]
    assert config.dataset in DATASETS, f"{config.dataset} not in {DATASETS.keys()}"
    assert config.model in MODELS, f"{config.model} not in {MODELS.keys()}"
    config.output_size = 10 if config.dataset == "cifar10" else 100
    # setup data module, model, and trainer
    datamodule = ImageDataModule(config)
    model = Model(config)
    callbacks = [
        # TQDMProgressBar(refresh_rate=0),
        ModelCheckpoint(
            filename="{epoch}_{avg_val_acc}",
            monitor="avg_val_acc",
            save_top_k=5,
            mode="max",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = WandbLogger(
        offline=not config.wandb,
        project=config.project_id,
        entity="chrisliu298",
        config=config,
    )
    trainer = Trainer(
        gpus=-1,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=1,
        benchmark=True,
        logger=logger,
    )
    wandb.log(
        {
            "train_size": len(datamodule.train_dataset),
            "val_size": len(datamodule.val_dataset),
            "test_size": len(datamodule.test_dataset),
        }
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    wandb.finish(quiet=True)


if __name__ == "__main__":
    main()
