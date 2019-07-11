import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

from utils.helpers import to_cuda, imshow
from utils.data_handler import InterpolationDataset
from model.split_model import SepConvNet

parser = argparse.ArgumentParser(description='SepConv Pytorch')
# params
parser.add_argument('--train_dir', type=str, default='data/dslf/train')
parser.add_argument('--val_dir', type=str, default='data/dslf/val')
parser.add_argument('--out_dir', type=str, default='outputs/output_dslf')
parser.add_argument('--load_model', type=str, default='weights/sepconv_weights')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

project_path = Path(__file__).parent.parent
train_dataset_path = Path(project_path/args.train_dir)
val_dataset_path = Path(project_path/args.val_dir)
# weights path
features_weight_path = Path(project_path/args.load_model/'features-lf.pytorch')
kernels_weight_path = Path(project_path/args.load_model/'kernels-lf.pytorch')


def train():
    # load the model and the weights
    model = to_cuda(SepConvNet(args.kernel))
    model.features.load_state_dict(torch.load(features_weight_path))
    model.subnet_kernel.load_state_dict(torch.load(kernels_weight_path))
    # get the dataset and data loader
    train_dataset = InterpolationDataset(train_dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    first_frame, gt_frame, sec_frame = next(iter(train_loader))
    print(first_frame.shape, gt_frame.shape, sec_frame.shape)
    imshow(first_frame[0], 'first_frame')
    imshow(sec_frame[0], 'sec_frame')
    # test the model
    out_frame = model(first_frame, sec_frame)
    print(f'Output shape: {out_frame.shape}')
    imshow(out_frame[0], 'out_frame')



if __name__ == '__main__':
    train()
