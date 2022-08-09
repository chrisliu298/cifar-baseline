import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_idx):
        super().__init__()
        assert block_idx <= 3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if block_idx == 0:
            self.maxpool = nn.Identity()
        if block_idx < 3:
            self.maxpool = nn.MaxPool2d(2)
        if block_idx == 3:
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        return self.maxpool(self.relu(self.bn(self.conv(x))))


class CNNBackbone(nn.Module):
    def __init__(self, num_classes=10, width_factor=64):
        super().__init__()
        self.channels = [3] + [width_factor * i for i in [1, 2, 4, 8]]
        self.num_classes = num_classes
        self.layer0 = BasicBlock(self.channels[0], self.channels[1], block_idx=0)
        self.layer1 = BasicBlock(self.channels[1], self.channels[2], block_idx=1)
        self.layer2 = BasicBlock(self.channels[2], self.channels[3], block_idx=2)
        self.layer3 = BasicBlock(self.channels[3], self.channels[4], block_idx=3)
        self.fc = nn.Linear(self.channels[-1], self.num_classes)

    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def Backbone(num_classes=10, width_factor=64):
    return CNNBackbone(num_classes, width_factor)
