import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.nn as nn
import math
import sys


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


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

# TODO: make the features extraction part of the network with the names matches the names in model.KernelEstimation


if __name__ == "__main__":
    frame0 = torch.randn((1, 3, 1280, 800))
    frame2 = torch.randn((1, 3, 1280, 800))
    x = to_cuda(torch.cat([frame0, frame2], 1))
    print(x.shape)

    basic1 = to_cuda(basic_block(6, 32))
    x = basic1(x)
    print('after basic 1, x has shape: ', x.shape)

    x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
    basic2 = to_cuda(basic_block(32, 64))
    x = basic2(x)
    print('after basic 2, x has shape: ', x.shape)

    x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
    basic3 = to_cuda(basic_block(64, 128))
    x = basic3(x)
    print('after basic 3, x has shape: ', x.shape)

    x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
    basic4 = to_cuda(basic_block(128, 256))
    x = basic4(x)
    print('after basic 4, x has shape: ', x.shape)

    x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
    basic5 = to_cuda(basic_block(256, 512))
    x = basic5(x)
    print('after basic 5, x has shape: ', x.shape)

    x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
    basic6 = to_cuda(basic_block(512, 512))
    x = basic6(x)
    print('after basic 6, x has shape: ', x.shape)

    x = to_cuda(upsampling_block(512))(x)
    basic7 = to_cuda(basic_block(512, 256))
    x = basic7(x)
    print('after first upsampling, x has shape: ', x.shape)

    x = to_cuda(upsampling_block(256))
    basic8 = to_cuda(basic_block(256, 128))
