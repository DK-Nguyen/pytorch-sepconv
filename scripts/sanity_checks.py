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
    print(f"----Length of features weight: {len(list(features_weight))}")
    for param_name, weight in features_weight.items():
        print(param_name, weight.shape)
    assert len(list(features_weight)) == 62, "The feature extraction part " \
                                             "has less than 62 params"
    print(f"----Length of kernels weight: {len(list(kernels_weight))}----", )
    for param_name, weight in kernels_weight.items():
        print(param_name, weight.shape)
    assert len(list(kernels_weight)) == 32, "The separated kernels part" \
                                            "has less than 32 params"


if __name__ == "__main__":
    reading_weights(features_weight_path, kernels_weight_path)