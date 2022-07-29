import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy

from models import *

MODELS = {
    "simplecnn": SimpleCNN5,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "preactresnet18": PreActResNet18,
    "preactresnet34": PreActResNet34,
    "preactresnet50": PreActResNet50,
    "preactresnet101": PreActResNet101,
    "preactresnet152": PreActResNet152,
    "wideresnet": WideResNet,
}


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = MODELS[self.config.model](num_classes=self.config.output_size)

    def forward(self, x):
        self.model(x)

    def evaluate(self, batch, stage=None):
        x, y = batch
        output = self.model(x)
        pred = torch.argmax(output, dim=1)
        loss = F.cross_entropy(output, y)
        acc = accuracy(pred, y)
        if stage:
            self.log(f"{stage}_loss", loss, logger=True)
            self.log(f"{stage}_acc", acc, logger=True)
        return loss, acc

    def on_train_start(self):
        # Log model parameters
        model_info = summary(self.model, input_size=(1, 3, 32, 32), verbose=0)
        self.log(
            "params",
            torch.tensor(model_info.total_params, dtype=torch.float32),
            logger=True,
        )

    def training_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "train")
        return {"loss": loss, "train_acc": acc}

    def training_epoch_end(self, outputs):
        acc = torch.stack([i["train_acc"] for i in outputs]).mean()
        loss = torch.stack([i["loss"] for i in outputs]).mean()
        self.log("avg_train_acc", acc, logger=True, prog_bar=True)
        self.log("avg_train_loss", loss, logger=True)

    def validation_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "val")
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([i["val_acc"] for i in outputs]).mean()
        loss = torch.stack([i["val_loss"] for i in outputs]).mean()
        self.log("avg_val_acc", acc, logger=True, prog_bar=True)
        self.log("avg_val_loss", loss, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.evaluate(batch, "test")
        return {"test_loss": loss, "test_acc": acc}

    def test_epoch_end(self, outputs):
        acc = torch.stack([i["test_acc"] for i in outputs]).mean()
        loss = torch.stack([i["test_loss"] for i in outputs]).mean()
        self.log("avg_test_acc", acc, logger=True, prog_bar=True)
        self.log("avg_test_loss", loss, logger=True)

    def configure_optimizers(self):
        if self.config.optimizer == "adamw":
            opt = optim.AdamW(
                self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd
            )
        elif self.config.optimizer == "sgd":
            opt = optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.wd,
            )
        elif self.config.optimizer == "adam":
            opt = optim.Adam(
                self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd
            )
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120, 160], gamma=0.2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1},
        }
