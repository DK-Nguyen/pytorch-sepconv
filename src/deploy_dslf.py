"""
This module takes a test folder (--test_dir with argparse) of images with names in form 0001.png, 0002.png ...
then gets the pretrained model (--weight_path) and do the interpolation for the images in the test folder
with distance specified by user (--distance).
The module first get the proper file names for interpolating according to the distance, copy these files into
the output folder, then load the model and do the interpolation. All images of the process will be saved in the
folder specified by --out_dir. Finally, the model calculates the min and mean PSNR of the interpolated folder
compared to the images in the ground truth folder.

Command to run the demo:
python deploy_dslf.py --mode one --test_dir data/dslf/demo --output_dir outputs/output_demo
"""

import torch
from pathlib import Path
import argparse
import shutil
import time
from tqdm import tqdm
import statistics
from datetime import datetime
import logging

from utils.helpers import get_file_names, to_cuda, imread, psnr, psnr_ycbcr, set_logger
from model.model import SepConvNet
from utils.data_handler import DeployDslfDataset

parser = argparse.ArgumentParser(description='Deploying SepConv Pretrained Model on DSLF Dataset')

parser.add_argument('--test_dir', type=str, default='data/dslf/test/icme2',
                    help='directory that contains input images')
parser.add_argument('--output_dir', type=str, default='outputs/output_deploy_dslf',
                    help='directory that contains output images')
parser.add_argument('--weight_path', type=str,
                    default='weights/deploy_weights/epoch002-batch_size1-08.13.2019_13-37-44_distance16-lr0.0005.pytorch')
parser.add_argument('--log_path', type=str, default='log_files/deploy_dslf.log',
                    help='the path to the log file that contains information for deploying dslf')
parser.add_argument('--mode', type=str, choices=['multiple', 'one'], default='multiple',
                    help='deploy on multiple images or one image, remember to change the in and out directory')
parser.add_argument('--distance', type=int, default=64,
                    help='distance is only used when "mode" is "multiple"')
parser.add_argument('--image_extension', type=str, default='.png',
                    help='extension of the images to deploy')

args = parser.parse_args()

# make the paths from args
now = datetime.now()
date_time = now.strftime("%m.%d.%Y")
project_dir = Path(__file__).parent.parent
test_dir = Path(project_dir / args.test_dir)

if args.mode == 'multiple':
    out_folder_name = date_time + '_' + test_dir.name + '_distance' + str(args.distance)
    out_dir = Path(project_dir / args.output_dir / out_folder_name)
else:
    out_dir = Path(project_dir / args.output_dir)

weight_path = Path(project_dir / args.weight_path)
log_path = Path(project_dir / args.log_path)


def prepare_output_dir(out_dir, test_dir):
    """
    Prepare the output directory when deploy the model on multiple images.
    :return: the file_names necessary to do the inference of the model on (see the function get_file_names in
             utils.helper for more information)
    """

    if not out_dir.exists():
        out_dir.mkdir()

    # get the necessary file names
    file_names = get_file_names(test_dir, args.distance, print_file_names=False)

    # copy the images in the firstIms into the output folder
    for name in file_names[1][0]:
        file_path = Path(test_dir / name)
        copy_to = Path(out_dir / name)
        shutil.copy(file_path, copy_to)

    # the firstIms list does not contain the last image,
    # so we need to also copy the last image of the secIms into the output folder
    last_im = file_names[1][1][-1]
    shutil.copy(Path(test_dir/last_im), Path(out_dir/last_im))

    return file_names


def get_model(weight_path):
    """
    :return: the sepconv model with loaded pretrained weight from the path given
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


def deploying(file_names, model, out_dir):
    """
    doing the interpolation
    for each round:
        for each image in the firstIms list:
            find the corresponding second image and output image name
            apply the model on the first image and second image, save the image with the output name
    """
    # get the logger
    total_interpolate_ims = 0
    for key, value in file_names.items():
        total_interpolate_ims += len(value[0])
        logging.info(f'Round #{key}, number of images to interpolate: {len(value[0])}')
        with tqdm(total=len(value[0])) as t:
            for idx, name in enumerate(value[0]):  # loop through the first list
                # get the proper paths
                first_im_path = Path(out_dir / name)
                sec_im_path = Path(out_dir / value[1][idx])
                out_im_path = Path(out_dir / value[2][idx])
                # print(f'Interpolating between {first_im_path} and {sec_im_path}')
                # read the images into 4-D Tensors and do the interpolation
                deploying = DeployDslfDataset(first_im_path, sec_im_path)
                deploying.run_model(model, out_im_path)
                # print(f'Writing the interpolated image to {out_im_path}')
                t.update()
    return total_interpolate_ims


def psnr_folders(test_dir, out_dir):
    """
    Find the min and mean PSNR of interpolated images in the output folder compared to the
    test folder.
    :return: min_psnr: the minimum psnr of 2 images in the 2 folders
             mean_psnr: the mean psnr of all the image pairs
    """
    logging.info('Evaluating the results...')
    psnr_errs = []
    psnr_ycbcr_errs = []
    for gt_im_path in test_dir.glob('*' + args.image_extension):
        interpolated_im_path = Path(out_dir / gt_im_path.name)
        # read the files into tensor
        gt_im = imread(gt_im_path, un_squeeze=False, un_normalize=True)
        interpolated_im = imread(interpolated_im_path, un_squeeze=False, un_normalize=True)
        # find psnr in RGB
        psnr_err = psnr(gt_im, interpolated_im)
        psnr_errs.append(psnr_err)
        # find psnr in ycbcr
        psnr_ycbcr_err = psnr_ycbcr(gt_im, interpolated_im)
        psnr_ycbcr_errs.append(psnr_ycbcr_err)

    psnr_errors = [f for f in psnr_errs if f != float("inf")]
    min_psnr = min(psnr_errors)
    psnr_ycbcr_errs = [i for i in psnr_ycbcr_errs if i != None]  # remove None from the list
    mean_psnr_ycbcr = statistics.mean(psnr_ycbcr_errs)
    return min_psnr, mean_psnr_ycbcr


if __name__ == '__main__':
    # get the logger file
    set_logger(log_path)

    if args.mode == 'multiple':
        start = time.time()
        model = get_model(weight_path=weight_path)
        file_names = prepare_output_dir(out_dir=out_dir, test_dir=test_dir)

        logging.info(f'--- Start Deploying the model on {args.test_dir}, distance {args.distance} ---')
        logging.info(f'Weight used: {args.weight_path}')
        logging.info(f'Interpolating and writing output images to {args.output_dir}')
        interpolate_ims = deploying(file_names, model, out_dir=out_dir)

        end = time.time()
        interpolate_time = end - start

        min_psnr, mean_psnr_ycbcr = psnr_folders(test_dir, out_dir)
        logging.info(f'Min PSNR: {min_psnr :.2f}. Mean PSNR in YCBCR: {mean_psnr_ycbcr :.2f}')
        logging.info(f'Total number of interpolated images: {interpolate_ims} ')
        logging.info(f'Time to interpolate {args.test_dir}: {interpolate_time:.2f}s')
        logging.info(f'--- Done ---')
    else:
        # if mode is 'one': deploying the model on only 1 image (for demo)
        model = get_model(weight_path=weight_path)
        logging.info(f'--- Start Demo ---')
        start = time.time()
        first_im_path = Path(test_dir / 'first.png')
        sec_im_path = Path(test_dir / 'sec.png')
        out_im_path = Path(out_dir / 'interpolated.png')
        # print(f'Interpolating between {first_im_path} and {sec_im_path}')
        # read the images into 4-D Tensors and do the interpolation
        deploying = DeployDslfDataset(first_im_path, sec_im_path)
        deploying.run_model(model, out_im_path)
        end = time.time()
        interpolate_time = end - start
        logging.info(f'Time to interpolate: {interpolate_time:.2f}s')
        logging.info(f'--- Done Demo! ---')


