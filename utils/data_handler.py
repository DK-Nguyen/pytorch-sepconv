import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from utils.helpers import to_cuda


class InterpolateDataset(Dataset):
    """

    """
    def __init__(self, input_dir, firstImsList, secImsList):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input_dir = input_dir
        self.firstImsList = firstImsList
        self.secImsList = secImsList

    def __getitem__(self, index):
        firstFrame = to_variable(self.transform(Image.open(os.path.join(self.input_dir, self.firstImsList[index]))))
        secFrame = to_variable(self.transform(Image.open(os.path.join(self.input_dir, self.secImsList[index]))))
        return firstFrame, secFrame

    def __len__(self):
        return len(self.firstImsList)