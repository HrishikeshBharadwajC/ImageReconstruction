import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, dilation= dilation, padding=padding, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class GeneratorASPP(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding="same")
        self.batch_norm1 = nn.BatchNorm2d(32)
        
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv2_3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv2_4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.adapt_pool2 = nn.AdaptiveMaxPool2d((128, 128))

        self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv3_2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv3_3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv3_4 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.adapt_pool3 = nn.AdaptiveMaxPool2d((64, 64))

        self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv4_3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv4_4 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.adapt_pool4 = nn.AdaptiveMaxPool2d((32, 32))
        
        self.conv5_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv5_2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv5_3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv5_4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm5 = nn.BatchNorm2d(128)
        self.adapt_pool5 = nn.AdaptiveMaxPool2d((16, 16))

        self.conv6_1 = SeparableConv2d(in_channels=128, out_channels=32, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv6_2 = SeparableConv2d(in_channels=128, out_channels=32, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv6_3 = SeparableConv2d(in_channels=128, out_channels=32, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv6_4 = SeparableConv2d(in_channels=128, out_channels=32, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm6 = nn.BatchNorm2d(128)
        self.adapt_pool6 = nn.AdaptiveMaxPool2d((8, 8))
        
        self.conv7_1 = SeparableConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv7_2 = SeparableConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv7_3 = SeparableConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv7_4 = SeparableConv2d(in_channels=128, out_channels=64, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm7 = nn.BatchNorm2d(256)
        self.adapt_pool7 = nn.AdaptiveMaxPool2d((4, 4))
        
        self.conv8_1 = SeparableConv2d(in_channels=256, out_channels=64, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv8_2 = SeparableConv2d(in_channels=256, out_channels=64, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv8_3 = SeparableConv2d(in_channels=256, out_channels=64, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv8_4 = SeparableConv2d(in_channels=256, out_channels=64, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm8 = nn.BatchNorm2d(256)
        self.adapt_pool8 = nn.AdaptiveMaxPool2d((2, 2))
        
        self.conv9_1 = SeparableConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, dilation=1, padding="same")
        self.conv9_2 = SeparableConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, dilation=2, padding="same")
        self.conv9_3 = SeparableConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, dilation=4, padding="same")
        self.conv9_4 = SeparableConv2d(in_channels=256, out_channels=128, kernel_size=4, stride=1, dilation=8, padding="same")
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.adapt_pool9 = nn.AdaptiveMaxPool2d((1, 1))

        self.deconv10 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.batch_norm10 = nn.BatchNorm2d(256)

        self.deconv11 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.batch_norm11 = nn.BatchNorm2d(256)

        self.deconv12 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.batch_norm12 = nn.BatchNorm2d(128)

        self.deconv13 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.batch_norm13 = nn.BatchNorm2d(128)
        
        self.deconv14 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.batch_norm14 = nn.BatchNorm2d(64)
        
        self.deconv15 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.batch_norm15 = nn.BatchNorm2d(64)
        
        self.deconv16 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.batch_norm16 = nn.BatchNorm2d(32)
        
        self.deconv17 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.batch_norm17 = nn.BatchNorm2d(32)
        
        self.deconv18 = nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.adaptmaxpool = nn.AdaptiveMaxPool2d((256, 256))
        self.adaptavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.batch_norm1(self.leakyReLU(self.conv1(x)))
        x2 = self.adapt_pool2(self.leakyReLU(self.batch_norm2(torch.cat([self.conv2_1(x1), self.conv2_2(x1), self.conv2_3(x1), self.conv2_4(x1)], dim=1))))
        x3 = self.adapt_pool3(self.leakyReLU(self.batch_norm3(torch.cat([self.conv3_1(x2), self.conv3_2(x2), self.conv3_3(x2), self.conv3_4(x2)], dim=1))))
        x4 = self.adapt_pool4(self.leakyReLU(self.batch_norm4(torch.cat([self.conv4_1(x3), self.conv4_2(x3), self.conv4_3(x3), self.conv4_4(x3)], dim=1))))
        x5 = self.adapt_pool5(self.leakyReLU(self.batch_norm5(torch.cat([self.conv5_1(x4), self.conv5_2(x4), self.conv5_3(x4), self.conv5_4(x4)], dim=1))))
        x6 = self.adapt_pool6(self.leakyReLU(self.batch_norm6(torch.cat([self.conv6_1(x5), self.conv6_2(x5), self.conv6_3(x5), self.conv6_4(x5)], dim=1))))
        x7 = self.adapt_pool7(self.leakyReLU(self.batch_norm7(torch.cat([self.conv7_1(x6), self.conv7_2(x6), self.conv7_3(x6), self.conv7_4(x6)], dim=1))))
        x8 = self.adapt_pool8(self.leakyReLU(self.batch_norm8(torch.cat([self.conv8_1(x7), self.conv8_2(x7), self.conv8_3(x7), self.conv8_4(x7)], dim=1))))
        x9 = self.adapt_pool9(self.leakyReLU(self.batch_norm9(torch.cat([self.conv9_1(x8), self.conv9_2(x8), self.conv9_3(x8), self.conv9_4(x8)], dim=1))))
        x10 = self.relu(self.batch_norm10(self.deconv10(x9)))
        x11 = self.relu(self.batch_norm11(self.deconv11(torch.cat([x10, x8], dim=1))))
        x12 = self.relu(self.batch_norm12(self.deconv12(torch.cat([x11, x7], dim=1))))
        x13 = self.relu(self.batch_norm13(self.deconv13(torch.cat([x12, x6], dim=1))))
        x14 = self.relu(self.batch_norm14(self.deconv14(torch.cat([x13, x5], dim=1))))
        x15 = self.relu(self.batch_norm15(self.deconv15(torch.cat([x14, x4], dim=1))))
        x16 = self.relu(self.batch_norm16(self.deconv16(torch.cat([x15, x3], dim=1))))
        x17 = self.relu(self.batch_norm17(self.deconv17(torch.cat([x16, x2], dim=1))))
        x18 = self.tanh(self.deconv18(torch.cat([x17, x1], dim=1)))
        return x18