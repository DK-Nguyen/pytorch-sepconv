"""
This module is used to deploy the model on a rectified dataset taken directly from the camera rig
The images will be in the folders Position02, Position05, Position06... with names like "
Position02_Camera01_rec02.png", "Position02_Camera02_rec01.png",...
We do interpolation with the number of output given by the user (default is 3) with this method:
in each position, we pick 2 consecutive images, say Position02_Camera01_rec02.png (I1) and
Position02_Camera02_rec01.png (I2)
Interpolate I1 and I2, get i2
Interpolate I1 and i2, get i1
Interpolate i2 and I2, get i3
"""

import argparse
from datetime import datetime
import time
from pathlib import Path
import torch
import shutil
import os
import logging

from model.model import SepConvNet
from utils.helpers import to_cuda, set_logger
from utils.data_handler import DeployCameraRigDataset


parser = argparse.ArgumentParser('Deploying SepConv Model on Camera Rig Dataset')
parser.add_argument('--data_dir', type=str, default='data/camera_rig',
                    help='the directory that contains the image folders')
parser.add_argument('--output_dir', type=str, default='outputs/output_deploy_camera_rig',
                    help='the output dir of the module')
parser.add_argument('--weight_path', type=str, default='weights/deploy_weights/network-l1',
                    help='the path to the pre-trained weights')
parser.add_argument('--num_output', type=int, default=3,
                    help='the number of desired interpolated images')
parser.add_argument('--image_extension', type=str, default='.png',
                    help='the image extension that you want to interpolate')
parser.add_argument('--resize', nargs=2, type=int, default=[1900, 1200],
                    help='(width, height) of the images to resize to')
parser.add_argument('--log_path', type=str, default='log_files/deploy_camera_rig.log',
                    help='the path to the log file that saves necessary information')

args = parser.parse_args()

# make the paths and folders from args
now = datetime.now()
date_time = now.strftime('%m.%d.%Y_%H-%M')
project_dir = Path(__file__).parent.parent
data_dir = Path(project_dir / args.data_dir)
if not Path(project_dir / args.output_dir).exists():
    Path(project_dir / args.output_dir).mkdir()
output_dir = Path(project_dir / args.output_dir / date_time)
weight_path = Path(project_dir / args.weight_path)
log_path = Path(project_dir / args.log_path)


def prepare_output_dir(data_dir, output_dir):
    """
    Prepare the output folder (make the directory, copy the data images into the output folders)
    :param output_dir: the output directory
    :return:
    """
    if not output_dir.exists():
        output_dir.mkdir()

    # copy the files from the data_dir into the output_dir
    for dir in os.listdir(data_dir):
        shutil.copytree(Path(data_dir / dir), Path(output_dir / dir))


def get_model(weight_path):
    """
    Get the model with pre-trained weights given by the path
    :param weight_path: the path to the weight (.pytorch file)
    :return: the model
    """
    sepconv_model = to_cuda(SepConvNet())
    if weight_path.name == 'network-l1':
        sepconv_model.features.load_state_dict(torch.load(Path(weight_path / 'features-l1.pytorch')))
        sepconv_model.subnet_kernel.load_state_dict(torch.load(Path(weight_path / 'kernels-l1.pytorch')))
    elif weight_path.name == 'network-lf':
        sepconv_model.features.load_state_dict(torch.load(Path(weight_path / 'features-lf.pytorch')))
        sepconv_model.subnet_kernel.load_state_dict(torch.load(Path(weight_path / 'kernels-lf.pytorch')))
    else:
        sepconv_model.load_state_dict(torch.load(weight_path).get('model_state_dict', ''))

    return sepconv_model


def interpolating(output_dir, resize=None):
    """
    Do the interpolation on the camera rig dataset.
    :param output_dir: the output directory that contains images copied from the dataset. Also it will
                        contain the interpolated images after we do interpolation.
    :return
    """
    model = get_model(weight_path)
    if args.num_output == 3:
        for position in sorted(os.listdir(output_dir)):
            position_path = Path(output_dir / position)
            print(f'Interpolating {position}')
            image_paths = (Path(position_path / image) for image in sorted(os.listdir(position_path)))
            it = iter(image_paths)
            for i in it:
                # get the names and the paths for the interpolated images
                # print(i.name, next(it).name)
                i1_name = i.stem + '_i1' + args.image_extension
                i1_path = Path(position_path / i1_name)
                i2_name = i.stem + '_i2' + args.image_extension
                i2_path = Path(position_path / i2_name)
                i3_name = i.stem + '_i3' + args.image_extension
                i3_path = Path(position_path / i3_name)
                output_paths = [i1_path, i2_path, i3_path]
                # print(i1_path, i2_path, i3_path)

                # now start interpolating and get the results
                interpolating_object = DeployCameraRigDataset(i, next(it), resize=resize)
                interpolating_object.interpolating(model, output_paths)


if __name__ == '__main__':
    set_logger(log_path)
    logging.info(f'--- Start deploying on the dataset {args.data_dir} ---')
    logging.info(f'Output Dir: {args.output_dir}')
    logging.info(f'weight used: {args.weight_path}')
    logging.info(f'number of output for a pair of images: {args.num_output}')

    start = time.time()
    prepare_output_dir(data_dir, output_dir)
    interpolating(output_dir, resize=None)  # resize can be None or tuple(args.resize)
    end = time.time()
    logging.info(f'--- Done. Execution time: {end - start :.2f}s ---')
