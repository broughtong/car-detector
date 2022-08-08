import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    # apply two convolutions, with active function
    def __init__(self, in_dim, out_dim):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)


class DoubleConvBN(nn.Module):
    # apply two convolutions, with active function
    def __init__(self, in_dim, out_dim):
        super(DoubleConvBN, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dim)
        )

    def forward(self,x):
        return self.double_conv(x)


class SmallerUnet(nn.Module):
    def __init__(self, num_classes=2):
        super(SmallerUnet, self).__init__()
        self.num_classes = num_classes

        # convolutions in contracting path
        self.left_conv1 = DoubleConv(3, 32)
        self.left_conv2 = DoubleConv(32, 64)
        self.left_conv3 = DoubleConv(64, 128)
        self.left_conv4 = DoubleConv(128, 256)
        self.left_conv5 = DoubleConv(256, 512)

        # max-pool layer in contracting path
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # up-sampling layers in expansive path
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        # convolutions in expansive path
        self.right_conv1 = DoubleConv(512, 256)
        self.right_conv2 = DoubleConv(256, 128)
        self.right_conv3 = DoubleConv(128, 64)
        self.right_conv4 = DoubleConv(64, 32)

        # last layer
        self.out_conv = nn.Conv2d(32, self.num_classes, kernel_size=1)

        # initialization
        self.weight_init()

    def forward(self, x):
        # contracting path
        x1 = self.left_conv1.forward(x)
        x2 = self.maxpool(x1)
        x2 = self.left_conv2.forward(x2)
        x3 = self.maxpool(x2)
        x3 = self.left_conv3.forward(x3)
        x4 = self.maxpool(x3)
        x4 = self.left_conv4.forward(x4)
        x5 = self.maxpool(x4)
        x5 = self.left_conv5.forward(x5)

        # expansive path
        x5 = self.upconv1(x5)
        x5 = torch.cat([x5, x4], dim=1)
        x5 = self.right_conv1.forward(x5)

        x5 = self.upconv2(x5)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.right_conv2.forward(x5)

        x5 = self.upconv3(x5)
        x5 = torch.cat([x5, x2], dim=1)
        x5 = self.right_conv3.forward(x5)

        x5 = self.upconv4(x5)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.right_conv4.forward(x5)

        # last layer
        x5 = self.out_conv(x5)
        return x5

    def weight_init(self):
        for layer in self.modules():
            if type(layer) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
                nn.init.xavier_uniform_(layer.weight)


class ZDUNet(nn.Module):
    def __init__(self, num_classes=2, num_channels=1):
        super(ZDUNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels

        # convolutions in contracting path
        self.left_conv1 = DoubleConvBN(self.num_channels, 8)
        self.left_conv2 = DoubleConvBN(8, 16)
        self.left_conv3 = DoubleConvBN(16, 32)
        self.left_conv4 = DoubleConvBN(32, 64)
        self.left_conv5 = DoubleConvBN(64, 128)
        self.left_conv6 = DoubleConvBN(128, 256)
        self.left_conv7 = DoubleConvBN(256, 512)
        self.left_conv8 = DoubleConvBN(512, 1024)

        # max-pool layer in contracting path
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # up-sampling layers in expansive path
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv6 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upconv7 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)

        # convolutions in expansive path
        self.right_conv1 = DoubleConvBN(1024, 512)
        self.right_conv2 = DoubleConvBN(512, 256)
        self.right_conv3 = DoubleConvBN(256, 128)
        self.right_conv4 = DoubleConvBN(128, 64)
        self.right_conv5 = DoubleConvBN(64, 32)
        self.right_conv6 = DoubleConvBN(32, 16)
        self.right_conv7 = DoubleConvBN(16, 8)

        # last layer
        self.out_conv = nn.Conv2d(8, self.num_classes, kernel_size=1)

        # initialization
        self.weight_init()

    def forward(self, x):
        # contracting path
        x1 = self.left_conv1.forward(x)
        x2 = self.maxpool(x1)
        x2 = self.left_conv2.forward(x2)
        x3 = self.maxpool(x2)
        x3 = self.left_conv3.forward(x3)
        x4 = self.maxpool(x3)
        x4 = self.left_conv4.forward(x4)
        x5 = self.maxpool(x4)
        x5 = self.left_conv5.forward(x5)
        x6 = self.maxpool(x5)
        x6 = self.left_conv6.forward(x6)
        x7 = self.maxpool(x6)
        x7 = self.left_conv7.forward(x7)
        x8 = self.maxpool(x7)
        x8 = self.left_conv8.forward(x8)

        # expansive path
        x8 = self.upconv1(x8)
        x8 = torch.cat([x8, x7], dim=1)
        x8 = self.right_conv1.forward(x8)

        x8 = self.upconv2(x8)
        x8 = torch.cat([x8, x6], dim=1)
        x8 = self.right_conv2.forward(x8)

        x8 = self.upconv3(x8)
        x8 = torch.cat([x8, x5], dim=1)
        x8 = self.right_conv3.forward(x8)

        x8 = self.upconv4(x8)
        x8 = torch.cat([x8, x4], dim=1)
        x8 = self.right_conv4.forward(x8)

        x8 = self.upconv5(x8)
        x8 = torch.cat([x8, x3], dim=1)
        x8 = self.right_conv5.forward(x8)

        x8 = self.upconv6(x8)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.right_conv6.forward(x8)

        x8 = self.upconv7(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.right_conv7.forward(x8)

        # last layer
        x8 = self.out_conv(x8)
        return x8

    def weight_init(self):
        for layer in self.modules():
            if type(layer) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d]:
                nn.init.xavier_uniform_(layer.weight)
