import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import sys
from model.layers.subnet_kernel import SubnetKernels
from model.layers.features_extraction import FeaturesExtraction
from model.layers.sepconv import FunctionSepconv


class SepConvNet(nn.Module):
    """
    The Separable Convolutional Network
    :param subnet_kernel_size: the number of in and out channels for the subnet kernels at the end of the network
    """
    def __init__(self, subnet_kernel_size=51):
        super(SepConvNet, self).__init__()
        self.kernel_size = subnet_kernel_size
        self.kernel_pad = subnet_kernel_size//2

        self.epoch = torch.tensor(0, requires_grad=False)
        self.features = FeaturesExtraction()
        self.subnet_kernel = SubnetKernels(subnet_kernel_size=subnet_kernel_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.modulePad = nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2])
        w0 = int(list(frame0.size())[3])
        h2 = int(list(frame2.size())[2])
        w2 = int(list(frame2.size())[3])
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False

        # pad the frames so that they height and width will be divisible by 2^5
        # because we down sampling and up sampling 5 times
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h))
            frame2 = F.pad(frame2, (0, 0, 0, pad_h))
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0))
            frame2 = F.pad(frame2, (0, pad_w, 0, 0))
            w_padded = True

        # extract the features with an auto-encoder
        tensorCombine = self.features(frame0, frame2)
        print('tensorCombine', tensorCombine.shape)

        # estimate a pair of 2D Convolution kernels K1 and K2 and
        # here, K1 = Vertical1 * Horizontal1 (*: convolution)
        #       K2 = Vertical2 * Horizontal2
        Vertical1, Horizontal1, Vertical2, Horizontal2 = self.subnet_kernel(tensorCombine)
        print('Vertical and Horizontal', Vertical1.shape, Horizontal1.shape, Vertical2.shape, Horizontal2.shape)

        # uses the pair of K1 and K2 to convolve with the input frames I1 and I2 to compute
        # the color of the output pixel I_hat
        tensorDot1 = FunctionSepconv()(self.modulePad(frame0), Vertical1, Horizontal1)
        tensorDot2 = FunctionSepconv()(self.modulePad(frame2), Vertical2, Horizontal2)
        print('tensorDot1 and 2', tensorDot1.shape, tensorDot2.shape)

        # I_hat(x,y) = K1(x,y)*P1(x,y) + K2(x,y)*P2(x,y)
        # where P1(x,y): the patches center at (x,y) in I1
        #       P2(x,y): the patches center at (x,y) in I2
        frame1 = tensorDot1 + tensorDot2
        print('frame1', frame1.shape)

        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]

        return frame1
