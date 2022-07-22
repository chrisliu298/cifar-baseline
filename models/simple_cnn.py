import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden_size = 64
        self.num_classes = num_classes
        layers = [
            nn.Conv2d(3, self.hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(),
        ]
        for i in range(3):
            layers.extend(
                [
                    nn.Conv2d(
                        self.hidden_size * (2**i),
                        self.hidden_size * (2 ** (i + 1)),
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm2d(self.hidden_size * (2 ** (i + 1))),
                    nn.ReLU(),
                    nn.MaxPool2d(2) if i < 2 else nn.AdaptiveAvgPool2d((1, 1)),
                ]
            )
        layers.extend([nn.Flatten(), nn.Linear(self.hidden_size * 8, self.num_classes)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def SimpleCNN5(num_classes=10):
    return CNN(num_classes)
