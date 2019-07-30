import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path, PurePosixPath
import cv2

from utils.helpers import to_cuda, imread, imwrite


class InterpolationDataset(Dataset):
    """
    Reads the images into tensors for training
    """

    def __init__(self, data_path, resize=None, im_extension='.png'):
        """
        :param data_path: the path to the dataset
        :param resize: the size to resize the image to
        """
        if resize is not None:
            self.transform = transforms.Compose([
                transforms.Resize(resize),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

        input_dir = Path(data_path/'input')
        gt_dir = Path(data_path/'gt')

        self.input_frame_paths = [Path(input_dir/f) for f in os.listdir(input_dir)]
        self.gt_frame_paths = [f.as_posix() for f in gt_dir.glob('*'+im_extension)]

        self.file_length = len(self.input_frame_paths)

    def __getitem__(self, index):
        """
        :return first_frame: the tensor for the first frame
                sec_frame: the tensor for the second frame
                gt_frame: the tensor for the ground truth frame
                gt_frame_name: the tuple that contains the names of the gt frames. Used to write output image with
                                corresponding names
        """
        # get the absolute (in string) for the frame paths
        first_frame_path = Path(self.input_frame_paths[index]/'first.png').as_posix()
        sec_frame_path = Path(self.input_frame_paths[index]/'sec.png').as_posix()
        gt_frame_path = self.gt_frame_paths[index]
        gt_frame_name = PurePosixPath(gt_frame_path).name

        first_frame = to_cuda(self.transform(cv2.imread(first_frame_path)))
        sec_frame = to_cuda(self.transform(cv2.imread(sec_frame_path)))
        gt_frame = to_cuda(self.transform(cv2.imread(gt_frame_path)))

        return first_frame, gt_frame, sec_frame, gt_frame_name

    def __len__(self):
        return self.file_length

    def get_path_lists(self):
        return self.input_frame_paths, self.gt_frame_paths


class DeployDslfDataset(Dataset):
    """
    The dataset that reads 2 images at a time, then interpolate and write the output image to disk
    Doing this saves memory.
    """
    def __init__(self, first_image_path, sec_image_path):
        self.first_im = imread(first_image_path)
        self.sec_im = imread(sec_image_path)

    def run_model(self, model, out_im_path):
        """
        Get the ouput from the model
        :param model: SepConv Model
        :param out_im_path: the path of the output image
        """
        with torch.no_grad():
            out_im = model(self.first_im, self.sec_im)
            imwrite(out_im, out_im_path, squeeze=True)
        # return out_im


class DeployCameraRigDataset(Dataset):
    """
    This dataset reads 2 images, then interpolate 3 images. Used in deploy_camera_rig.py
    """
    def __init__(self, first_im_path, sec_im_path, resize=None):
        """
        Constructor
        :param first_im_path: (pathlib ojbect) the path to the first image
        :param sec_im_path: (pathlib object) the path to the second image
        """
        self.first_im_path = first_im_path
        self.sec_im_path = sec_im_path
        self.first_im = imread(first_im_path, resize=resize)
        self.sec_im = imread(sec_im_path, resize=resize)

    def interpolating(self, model, output_im_paths):
        """
        Do interpolating on the 2 input images
        :param model: the SepConv Model
        :param output_im_paths: the list that contains the paths of the output images
                                e.g. [i1_path, i2_path, i3_path]
        """
        i1_path, i2_path, i3_path = output_im_paths
        print(f'Interpolating between {self.first_im_path.name} and {self.sec_im_path.name}')
        with torch.no_grad():
            i2 = model(self.first_im, self.sec_im)
            imwrite(i2, i2_path, squeeze=True)
            # print(f'writing {i2_path.name} to disk, '
            #       f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')
            i1 = model(self.first_im, i2)
            imwrite(i1, i1_path, squeeze=True)
            # print(f'writing {i1_path.name} to disk, '
            #       f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')
            i3 = model(i2, self.sec_im)
            imwrite(i3, i3_path, squeeze=True)
            # print(f'writing {i3_path.name} to disk, '
            #       f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')


