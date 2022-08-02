import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchinfo import summary
from torchmetrics.functional import accuracy

from models import *

MODELS = {
    "backbone": Backbone,
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "resnet101": ResNet101,
    "resnet152": ResNet152,
    "preact-resnet18": PreActResNet18,
    "preact-resnet34": PreActResNet34,
    "preact-resnet50": PreActResNet50,
    "preact-resnet101": PreActResNet101,
    "preact-resnet152": PreActResNet152,
    "resnext29-2x64d": ResNeXt29_2x64d,
    "resnext29-4x64d": ResNeXt29_4x64d,
    "resnext29-8x64d": ResNeXt29_8x64d,
    "resnext29-32x4d": ResNeXt29_32x4d,
    "wideresnet": WideResNet,
}


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = MODELS[self.config.model](num_classes=self.config.num_classes)

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
        # log model parameters
        model_info = summary(self.model, input_size=(1, 3, 32, 32), verbose=0)
        self.log("params", float(model_info.total_params), logger=True)
        # log data split sizes
        datamodule = self.trainer.datamodule
        self.log("train_size", float(len(datamodule.train_dataset)), logger=True)
        self.log("val_size", float(len(datamodule.val_dataset)), logger=True)
        self.log("test_size", float(len(datamodule.test_dataset)), logger=True)

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
        param_groups = self.configure_parameter_groups(self.model)
        if self.config.optimizer == "adamw":
            opt = optim.AdamW(param_groups, lr=self.config.lr)
        elif self.config.optimizer == "sgd":
            opt = optim.SGD(param_groups, lr=self.config.lr, momentum=0.9)
        elif self.config.optimizer == "adam":
            opt = optim.Adam(param_groups, lr=self.config.lr)
        sch = optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 80], gamma=0.2)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch", "frequency": 1},
        }

    def configure_parameter_groups(self, model):
        """
        Code adapted from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Conv2d, nn.Linear, nn.BatchNorm2d)
        blacklist_weight_modules = ()
        for module_name, module in model.named_modules():
            for param_name, _ in module.named_parameters():
                if len(module_name) == 0:
                    continue
                name = f"{module_name}.{param_name}"
                if name.endswith("bias"):
                    no_decay.add(name)
                elif name.endswith("weight") and isinstance(
                    module, whitelist_weight_modules
                ):
                    decay.add(name)
                elif name.endswith("weight") and isinstance(
                    module, blacklist_weight_modules
                ):
                    no_decay.add(name)
        param_dict = {
            param_name: param for param_name, param in model.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )
        decay = sorted(list(decay))
        no_decay = sorted(list(no_decay))
        # print("Decay:", decay)
        # print("No decay:", no_decay)
        return [
            {
                "params": [param_dict[pn] for pn in decay],
                "weight_decay": self.config.wd,
            },
            {"params": [param_dict[pn] for pn in no_decay], "weight_decay": 0.0},
        ]
