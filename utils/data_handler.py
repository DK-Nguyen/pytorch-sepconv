import os
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path, PurePosixPath
import cv2
from utils.helpers import to_cuda


class InterpolationDataset(Dataset):
    """
    Reads the images into tensors
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


