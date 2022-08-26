from torch import Tensor
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, channels=1, sigmoid=False) -> None:
        super(Discriminator, self).__init__()
        layers = [
            nn.Conv2d(channels, 64, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, 1, (4, 4), (1, 1), (0, 0), bias=True),
        ]
        if sigmoid:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

            
    def forward(self, x: Tensor) -> Tensor:
        out = self.main(x)
        out = torch.flatten(out, 1)
        return out

class Generator(nn.Module):
    def __init__(self, channels=1) -> None:
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 512 * 4 * 4)
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            nn.ConvTranspose2d(64, channels, (4, 4), (2, 2), (1, 1), bias=True), 

            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = x.view(x.size(0), 512, 4, 4)
        out = self.cnn(x)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)