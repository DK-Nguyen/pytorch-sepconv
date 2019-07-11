import math
import os
import torch
import numpy as np
from matplotlib import pyplot as plt


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def get_file_names(input_dir, distance, print_file_names=False):
    """
    Get the lists of corresponding file names we need to interpolate images (used to run the pre-trained weight
    on a dataset that consists images with names in form '0001.extension', '0002.extension'...)

    :param input_dir: the path to the folder that contains ground truth images
    :param distance: the distance between 2 ground truth images that we want to get the interpolated image of them
    :param print_file_names: if we want to print the file_names after getting them, set it to 'True'
    :return file_names - if distance = 4, we have 2 rounds, the return variable is a map that looks like this:
                        {1: ([001.png, 0005.png..., 189.png],
                            [0005.png, 0009.png..., 193.png],
                            [0003i.png, 0007i.png, ..., 191i.png])
                         2: ([0001.png, 0003i.png,..., 191i.png]
                             [0003i.png, 0005.png,..., 193.png]
                             [0002ii.png, 0004ii.png,..., 192ii.png])
                         final: [0001.png, 0002ii.png, 0003i.png, 0004ii.png, 0005.png, ..., 192ii.png, 193.png]
                        }

                        Explanation:
                        Round 1: Interpolating between 1 & 5 to get 3i, 5 & 9 to get 7i, ...
                                 The first_ims list is  [0001.png, 0005.png, 0009.png, ..., 189.png]
                                 The sec_ims list is    [0005.png, 0009.png, 0013.png, ..., 193.png]
                                 The output_ims list is [0003i.png, 0007i.png, 0011i.png, ..., 191i.png]
                        Round 2. Interpolating between 1 & 3i to get 2ii, 3i & 5 to get 4ii, ...
                                 The first_ims list is  [0001.png, 0003i.png, 0005.png, ..., 191i.png]
                                 The sec_ims list is    [0003i.png, 0005.png, 0007i.png, ..., 193.png]
                                 The output_ims list is [0002ii.png, 0004ii.png, 0006ii.png, ..., 192ii.png]
                        The final names: [0001.png, 0002ii.png, 0003i.png, 0004ii.png, 0005.png..., 192ii.png, 193.png]
    """

    number_of_rounds = int(math.log2(distance))
    assert distance in range(2, 65), "distance must be in the range of [2, 64]"
    assert number_of_rounds % 1 == 0, "distance must be a power of 2 e.g. 2, 4, 8, 16..."

    gt_files = []  # ground truth files
    for folders, sub_folders, files in os.walk(input_dir):
        if '.ipynb_checkpoints' not in folders:
            gt_files[:] = [f for f in files if not f.startswith("_")]  # get all the names of the files in inputDir

    file_names = {}
    for round_number in range(1, number_of_rounds + 1):
        file_names[round_number] = ()
        if round_number == 1:
            first_ims = gt_files[0::distance][:-1]
            sec_ims = gt_files[distance::distance]
            output_ims = gt_files[int(distance / 2)::distance]
            # put 'i' into the names of interpolated files e.g. 0003.png -> 0003i.png
            output_ims[:] = [(name.split('.')[0] + 'i' + '.' + name.split('.')[1]) for name in output_ims]
            assert len(first_ims) == len(sec_ims), "Lengths of first list and second list are different"
            assert len(first_ims) == len(output_ims), "Lengths of first list and output list are different"
            file_names[round_number] += (first_ims, sec_ims, output_ims)
        else:
            # From round 2, the first_ims list is concatenated from the first_ims & output_ims of the previous round.
            # Similarly, the sec_ims list is concatenated from the sec_ims & output_ims of the previous round.
            first_ims = sorted(file_names[round_number - 1][0] + file_names[round_number - 1][2])
            sec_ims = sorted(file_names[round_number - 1][1] + file_names[round_number - 1][2])
            output_ims = gt_files[int(distance / (2 ** round_number))::int(distance / (2 ** (round_number - 1)))]
            output_ims[:] = [(name.split('.')[0] + 'i'*round_number + '.' + name.split('.')[1]) for name in output_ims]
            assert len(first_ims) == len(sec_ims), \
                print(f"Lengths of first list and second list are different: {len(first_ims)} vs {len(sec_ims)}")
            assert len(first_ims) == len(output_ims), \
                print(f"Lengths of first list and output list are different: {len(first_ims)} vs {len(output_ims)}")
            file_names[round_number] += (first_ims, sec_ims, output_ims)

    # final output is concatenated from all of the round's outputs, 
    # the first images and the last image of the first round.
    # the length must be 193
    file_names['final'] = []
    for round_number in range(1, number_of_rounds + 1):  # concatenating the outputs of all rounds
        file_names['final'] += file_names[round_number][2]
    file_names['final'] += file_names[1][0]  # concatenating the first images of the first round
    file_names['final'].append(file_names[1][1][-1])  # concatenating the the last image.
    file_names['final'] = sorted(file_names['final'])
    assert len(file_names['final']) == 193, print(f"Length of the final list is {len(file_names['final'])}, not 193")

    # print the file_names
    if print_file_names:
        for round_num, fileNameLists in file_names.items():
            if round_num != 'final':
                print(f'Round Number {round_num}, sizes of the lists: '
                      f'{len(fileNameLists[0])}, {len(fileNameLists[1])}, {len(fileNameLists[2])}')
                print(fileNameLists[0])
                print('Second Images Names:', fileNameLists[1])
                print('Interpolated Images Names:', fileNameLists[2])
                print()
            else:
                print('Final Names:', fileNameLists)

    return file_names


def imshow(image, title=None, ax=None, normalize=False):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if torch.cuda.is_available():
        image = image.cpu().detach()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.title(title)

    return ax