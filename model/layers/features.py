import torch
import torch.optim as optim
from torch.nn import functional as F
import math
import sys

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
       pass
