import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels * 2,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=4,
            stride=1,
            padding=1,
            bias=False,
        )
        self.batch_norm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1
        )

        self.leakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        x1 = self.leakyReLU(self.conv1(input))
        x2 = self.leakyReLU(self.batch_norm2(self.conv2(x1)))
        x3 = self.leakyReLU(self.batch_norm3(self.conv3(x2)))
        x4 = self.leakyReLU(self.batch_norm4(self.conv4(x3)))
        x5 = self.conv5(x4)
        return x5
