"""
This Module is used to split the pre-trained weights into 2 parts (features extraction and
kernels) for transfer learning.
"""

import torch
from pathlib import Path
from collections import OrderedDict

weight_folder = Path(__file__).parent.parent/'weights'/'sepconv_weights'
full_weight = (weight_folder/'network-lf.pytorch').resolve()
feature_weight = Path(weight_folder/'features-lf.pytorch')
kernel_weight = Path(weight_folder/'kernels-lf.pytorch')


def splitting_weights(full_weight_path, feature_weight_path, kernel_weight_path):
    full_weights = torch.load(full_weight_path)
    feature_weights = OrderedDict()  # the weights for the feature extraction part of the network
    kernel_weights = OrderedDict()  # the weights for the sepconv kernels part of the network

    counter = 0
    for idx, value in full_weights.items():
        if counter < 62:
            feature_weights[idx] = value
        else:
            kernel_weights[idx] = value
        counter += 1

    # print the weights of the feature extraction
    print('Feature Extraction Weights: ')
    for idx, value in feature_weights.items():
        print(idx, value.shape)

    # print the weights of the sepconv kernels part
    print('SepConv Kernels Weights: ')
    for idx, value in kernel_weights.items():
        print(idx, value.shape)

    # save the dicts into .pytorch files
    torch.save(feature_weights, feature_weight_path)
    torch.save(kernel_weights, kernel_weight_path)


if __name__ == '__main__':
    splitting_weights(full_weight, feature_weight, kernel_weight)

