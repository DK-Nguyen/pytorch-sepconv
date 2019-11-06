import torch
from torch.nn import functional as F
import torch.nn as nn


def basic_block(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False)
    )


def upsampling_block(channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False)
    )


class FeaturesExtraction(nn.Module):
    """
    The feature extraction part of the SepConv Network
    """

    def __init__(self):
        super(FeaturesExtraction, self).__init__()
        self.moduleConv1 = basic_block(6, 32)

        self.modulePool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleConv2 = basic_block(32, 64)

        self.modulePool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleConv3 = basic_block(64, 128)

        self.modulePool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleConv4 = basic_block(128, 256)

        self.modulePool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleConv5 = basic_block(256, 512)

        self.modulePool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.moduleDeconv5 = basic_block(512, 512)

        self.moduleUpsample5 = upsampling_block(512)
        self.moduleDeconv4 = basic_block(512, 256)

        self.moduleUpsample4 = upsampling_block(256)
        self.moduleDeconv3 = basic_block(256, 128)

        self.moduleUpsample3 = upsampling_block(128)
        self.moduleDeconv2 = basic_block(128, 64)

        self.moduleUpsample2 = upsampling_block(64)

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)

        tensorPool1 = self.modulePool1(tensorConv1)
        tensorConv2 = self.moduleConv2(tensorPool1)

        tensorPool2 = self.modulePool2(tensorConv2)
        tensorConv3 = self.moduleConv3(tensorPool2)

        tensorPool3 = self.modulePool3(tensorConv3)
        tensorConv4 = self.moduleConv4(tensorPool3)

        tensorPool4 = self.modulePool4(tensorConv4)
        tensorConv5 = self.moduleConv5(tensorPool4)

        tensorPool5 = self.modulePool5(tensorConv5)
        tensorDeconv5 = self.moduleDeconv5(tensorPool5)

        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)
        tensorCombine = tensorUpsample5 + tensorConv5
        tensorDeconv4 = self.moduleDeconv4(tensorCombine)

        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)
        tensorCombine = tensorUpsample4 + tensorConv4
        tensorDeconv3 = self.moduleDeconv3(tensorCombine)

        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        tensorCombine = tensorUpsample3 + tensorConv3
        tensorDeconv2 = self.moduleDeconv2(tensorCombine)

        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        tensorCombine = tensorUpsample2 + tensorConv2

        return tensorCombine

