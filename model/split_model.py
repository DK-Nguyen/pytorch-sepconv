import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import layers.sepconv
from layers import features


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

