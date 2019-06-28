import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from pathlib import Path

weights_path = Path(__file__).parent.parent/'weights'
print('Weights Folder Path:', weights_path)
features_weight_path = Path(weights_path/'sepconv_weights/features-lf.pytorch')
kernels_weight_path = Path(weights_path/'sepconv_weights/kernels-lf.pytorch')


def reading_weights(features_path, kernels_path):
    features_weight = torch.load(features_path)
    kernels_weight = torch.load(kernels_path)
    print('Length of features weight:', len(list(features_weight)))
    print('Length of kernels weight:', len(list(kernels_weight)))

    # i = 0
    # for idx, value in kernels_weight.items():
    #     print(idx, value.shape)
    #     print(value)
    #     if i == 1:
    #         break
    #     i += 1


# def reading_models()


if __name__ == "__main__":
    reading_weights(features_weight_path, kernels_weight_path)