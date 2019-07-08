import torch
import torch.nn as nn


def subnet(subnet_kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(in_channels=64, out_channels=subnet_kernel_size, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(in_channels=subnet_kernel_size, out_channels=subnet_kernel_size,
                  kernel_size=3, stride=1, padding=1)
    )


class SubnetKernels(torch.nn.Module):
    def __init__(self, subnet_kernel_size):
        super(SubnetKernels, self).__init__()
        self.subnet_kernel_size = subnet_kernel_size
        self.moduleVertical1 = subnet(self.subnet_kernel_size)
        self.moduleVertical2 = subnet(self.subnet_kernel_size)
        self.moduleHorizontal1 = subnet(self.subnet_kernel_size)
        self.moduleHorizontal2 = subnet(self.subnet_kernel_size)

    def forward(self, tensorCombine):
        Vertical1 = self.moduleVertical1(tensorCombine)
        Vertical2 = self.moduleVertical2(tensorCombine)
        Horizontal1 = self.moduleHorizontal1(tensorCombine)
        Horizontal2 = self.moduleHorizontal2(tensorCombine)

        return Vertical1, Horizontal1, Vertical2, Horizontal2
