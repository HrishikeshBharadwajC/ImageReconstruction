import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1
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
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm7 = nn.BatchNorm2d(512)

        # Bottleneck
        self.conv8 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1
        )

        # Decoder
        self.deconv9 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm9 = nn.BatchNorm2d(512)

        self.deconv10 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm10 = nn.BatchNorm2d(512)

        self.deconv11 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm11 = nn.BatchNorm2d(512)

        self.deconv12 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm12 = nn.BatchNorm2d(512)

        self.deconv13 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm13 = nn.BatchNorm2d(256)

        self.deconv14 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm14 = nn.BatchNorm2d(128)

        self.deconv15 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.batch_norm15 = nn.BatchNorm2d(64)

        self.deconv16 = nn.ConvTranspose2d(
            in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1
        )

        self.dropout = nn.Dropout(0.5)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.leakyReLU(self.conv1(x))
        x2 = self.leakyReLU(self.batch_norm2(self.conv2(x1)))
        x3 = self.leakyReLU(self.batch_norm3(self.conv3(x2)))
        x4 = self.leakyReLU(self.batch_norm4(self.conv4(x3)))
        x5 = self.leakyReLU(self.batch_norm5(self.conv5(x4)))
        x6 = self.leakyReLU(self.batch_norm6(self.conv6(x5)))
        x7 = self.leakyReLU(self.batch_norm7(self.conv7(x6)))
        bottleneck = self.relu(self.conv8(x7))
        x9 = self.dropout(self.relu(self.batch_norm9(self.deconv9(bottleneck))))
        x10 = self.dropout(
            self.relu(self.batch_norm10(self.deconv10(torch.cat([x9, x7], dim=1))))
        )
        x11 = self.dropout(
            self.relu(self.batch_norm11(self.deconv11(torch.cat([x10, x6], dim=1))))
        )
        x12 = self.relu(self.batch_norm12(self.deconv12(torch.cat([x11, x5], dim=1))))
        x13 = self.relu(self.batch_norm13(self.deconv13(torch.cat([x12, x4], dim=1))))
        x14 = self.relu(self.batch_norm14(self.deconv14(torch.cat([x13, x3], dim=1))))
        x15 = self.relu(self.batch_norm15(self.deconv15(torch.cat([x14, x2], dim=1))))
        x13 = self.tanh(self.deconv16(torch.cat([x15, x1], dim=1)))
        return x13
