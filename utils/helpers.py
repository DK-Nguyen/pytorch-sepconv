import math
import os
import torch
from torchvision import transforms
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv
from pathlib import Path
import json


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def get_file_names(input_dir, distance, print_file_names=False):
    """
    Get the lists of corresponding image names we need to interpolate in a folder that contains images with
    names in form 0001.png, 0002.png ...

    :param input_dir: the path to the folder that contains ground truth images
    :param distance: the distance between 2 ground truth images
                     distance must be 2^r (r = [2, 3, 4 ... 6])
    :param print_file_names: if we want to print the file_names after getting them, set it to 'True'
    :return file_names (a dictionary) - if distance = 4, we have 2 rounds, the returned dict looks like this:
                        {1: ([001.png, 0005.png..., 189.png],
                            [0005.png, 0009.png..., 193.png],
                            [0003i.png, 0007.png, ..., 191.png])
                         2: ([0001.png, 0003.png,..., 191.png]
                             [0003.png, 0005.png,..., 193.png]
                             [0002.png, 0004.png,..., 192.png])
                        }

                        Explanation:
                        Round 1: Interpolating between 1 & 5 to get 3i, 5 & 9 to get 7i, ...
                                 The first_ims list is  [0001.png, 0005.png, 0009.png, ..., 189.png]
                                 The sec_ims list is    [0005.png, 0009.png, 0013.png, ..., 193.png]
                                 The output_ims list is [0003.png, 0007.png, 0011.png, ..., 191.png]
                        Round 2. Interpolating between 1 & 3 to get 2, 3 & 5 to get 4, ...
                                 The first_ims list is  [0001.png, 0003.png, 0005.png, ..., 191.png]
                                 The sec_ims list is    [0003.png, 0005.png, 0007.png, ..., 193.png]
                                 The output_ims list is [0002.png, 0004.png, 0006.png, ..., 192.png]
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
            output_ims[:] = [(name.split('.')[0] + '.' + name.split('.')[1]) for name in output_ims]
            assert len(first_ims) == len(sec_ims), "Lengths of first list and second list are different"
            assert len(first_ims) == len(output_ims), "Lengths of first list and output list are different"
            file_names[round_number] += (first_ims, sec_ims, output_ims)
        else:
            # From round 2, the first_ims list is concatenated from the first_ims & output_ims of the previous round.
            # Similarly, the sec_ims list is concatenated from the sec_ims & output_ims of the previous round.
            first_ims = sorted(file_names[round_number - 1][0] + file_names[round_number - 1][2])
            sec_ims = sorted(file_names[round_number - 1][1] + file_names[round_number - 1][2])
            output_ims = gt_files[int(distance / (2 ** round_number))::int(distance / (2 ** (round_number - 1)))]
            output_ims[:] = [(name.split('.')[0] + '.' + name.split('.')[1]) for name in output_ims]
            assert len(first_ims) == len(sec_ims), \
                print(f"Lengths of first list and second list are different: {len(first_ims)} vs {len(sec_ims)}")
            assert len(first_ims) == len(output_ims), \
                print(f"Lengths of first list and output list are different: {len(first_ims)} vs {len(output_ims)}")
            file_names[round_number] += (first_ims, sec_ims, output_ims)

    # # final output is concatenated from all of the round's outputs,
    # # the first images and the last image of the first round.
    # # the length must be 193
    # final_names = []
    # for round_number in range(1, number_of_rounds + 1):  # concatenating the outputs of all rounds
    #     final_names += file_names[round_number][2]
    # final_names += file_names[1][0]  # concatenating the first images of the first round
    # final_names.append(file_names[1][1][-1])  # concatenating the the last image.
    # final_names = sorted(final_names)
    # assert len(final_names) == 193, print(f"Length of the final list is {len(final_names)}, not 193")

    # print the file_names
    if print_file_names:
        print(f'--- File names with Distance {distance}: ----')
        for round_num, fileNameLists in file_names.items():
            print(f'Round #{round_num}, sizes of the lists: '
                  f'{len(fileNameLists[0])}, {len(fileNameLists[1])}, {len(fileNameLists[2])}')
            print('First Images Names', fileNameLists[0])
            print('Second Images Names:', fileNameLists[1])
            print('Interpolated Images Names:', fileNameLists[2])
            print()

    return file_names


def imshow(tensor, title=None, ax=None, normalize=False):
    """
    Show a 3D tensor as an image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if torch.cuda.is_available():
        tensor = tensor.cpu().detach()
    tensor = tensor.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor = std * tensor + mean
        tensor = np.clip(tensor, 0, 1)

    ax.imshow(tensor)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    plt.title(title)

    return ax


def imwrite(tensor, output_path, un_normalize=True, squeeze=False):
    """
    Write a 3D tensor into image on disk.
    :param tensor: (torch.tensor) the 3D or 4D tensor of an image
           output_path: (pathlib object or str) the path to save to image to
           un_normalize: (bool) if True, convert the tensor from range [0,1] to range [0,255]
           squeeze: (bool) if True, squeeze to remove the fake dimension of the tensor
    """
    if torch.cuda.is_available():
        tensor = tensor.cpu().detach()
    if squeeze:
        tensor = tensor.squeeze(0)
    tensor = tensor.numpy().transpose((1, 2, 0))
    if un_normalize:
        tensor = tensor * 255  # convert to [0, 255] range
    if type(output_path) is not str:  # convert to string if the path is pathlib.Path
        output_path = output_path.as_posix()

    cv2.imwrite(output_path, tensor)


def imread(im_path, un_squeeze=True, un_normalize=False, resize=None):
    """
    Read the image from the path given.
    :param im_path: the path to the image
    :param un_squeeze: boolean parameter to tell if we need to unsqeeze the tensor (make a fake dimension to put
                      into the model)
           un_normalize: if True, convert the image to [0, 255] range
           resize: (width, height) resize to a desirable size if provided. Used when 2 images have different sizes
    :return: if unsqueeze is True, the 4D tensor with shape (1, #channels, height, width) that represents the image
            else it is the 3D tensor
    """
    if type(im_path) is not str:
        im_path = im_path.as_posix()
    image = cv2.imread(im_path)
    if resize is not None:
        image = cv2.resize(image, resize)
    transform = transforms.Compose([transforms.ToTensor()])
    image = to_cuda(transform(image))
    if un_squeeze:
        image = image.unsqueeze(0)
    if un_normalize:
        image = image*255
    return image


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)


def rgb2ycbcr(im_rgb):
    """
    Convert the image from rgb to ycbcr color space. Used in the function psnr_ycbcr() below
    :param im_rgb: 3D numpy array that represents an image in RGB color space
    :return: the image in ycbcr color space
    """
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    im_ycbcr[:, :, 0] = (im_ycbcr[:, :, 0]*(235-16)+16)/255.0 #to [16/255, 235/255]
    im_ycbcr[:, :, 1:] = (im_ycbcr[:, :, 1:]*(240-16)+16)/255.0 #to [16/255, 240/255]
    return im_ycbcr


def psnr_ycbcr(gt_im, interpolated_im):
    """
    Find the psnr of 2 images in ycbcr color space. Only apply on the first channel of the images
    :param gt_im: the ground truth image in RGB color space in 3D tensor
    :param interpolated_im: the interpolated image in RGB color space in 3D tensor
    :return: the PSNR of the first channel of 2 images in ycbcr color space.
    """
    # convert the images from tensor to numpy arrays
    if torch.cuda.is_available():
        gt_im = gt_im.cpu().detach()
        gt_im = gt_im.permute(1, 2, 0).numpy()
        interpolated_im = interpolated_im.cpu().detach()
        interpolated_im = interpolated_im.permute(1, 2, 0).numpy()

    # convert the images into ycbcr color space, get only the first channels
    gt_im_ycbcr = rgb2ycbcr(gt_im)[:, :, 0]
    interpolated_im_ycbcr = rgb2ycbcr(interpolated_im)[:, :, 0]
    # find the psnr of the first channels
    e = np.abs(gt_im_ycbcr - interpolated_im_ycbcr) ** 2
    mse = np.sum(e) / e.size
    if mse > 0.001:  # mse should not be zero
        psnr_err = 10 * np.log10(255**2 / mse)
        return psnr_err
    else:
        pass


def psnr(gt_im, interpolated_im):
    """
    Find the PSNR between a ground truth image and an interpolated image.
    :param gt_im: the ground truth image in tensor
    :param interpolated_im: the interpolated image in tensor
    :return: float
    """
    e = torch.abs(gt_im - interpolated_im) ** 2
    mse = torch.sum(e) / e.numel()
    psnr_err = 10 * torch.log10(torch.tensor(255) * torch.tensor(255) / mse)

    return psnr_err.item()


def save_csv(save_to, **kwargs):
    """

    :param save_to: the path to save the csv file to
    :param kwargs: the key-value pairs to be saved into the csv file
    :return:
    """
    print(f'--- Saving log information to {save_to} ---')
    with open(save_to, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=kwargs.keys())
        writer.writeheader()
        writer.writerow(dict(kwargs))


class Param:
    """
    Class that reads hyper-parameters from a json file

    Example:
    params = Param(json_path)
    print(params.learning_rate)
    >> params.learning_rate = 0.0005
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params =json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


if __name__ == '__main__':
    project_path = Path(__file__).parent.parent
    im_path = Path(project_path / 'data/camera_rig/Position02/Position02_Camera10_rec01.png')
    output_path = Path(project_path / 'outputs/experiment.png')
    im = imread(im_path)
    im_resized = imread(im_path, resize=(1900, 1200))
    print(im.shape)
    print(im_resized.shape)
    imwrite(im_resized, output_path, squeeze=True)

