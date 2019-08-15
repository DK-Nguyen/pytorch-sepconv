"""
This module gets the training folder (specified by --train_dir) and val folder (--val_dir),
load the pretrained_weights from the directory specified by --load_model, do the transfer learning
(fine tune the pretrained weights on the last part of the network - the subnet_kernel) on the train and val
folder, then save the weights to the folder specified by --save_weights.
The output images will be saved to the --out_dir folder.
"""

import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from pathlib import Path
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
import logging

from model.model import SepConvNet
from utils.helpers import to_cuda, imwrite, plot_losses, psnr, Param, set_logger
from utils.data_handler import InterpolationDataset


parser = argparse.ArgumentParser(description='SepConv Pytorch')
# params
parser.add_argument('--train_dir', type=str, default='data/dslf/train_16',
                    help='the folder that contains training images')
parser.add_argument('--val_dir', type=str, default='data/dslf/val_16',
                    help='the folder that contains validation images')
parser.add_argument('--out_dir', type=str, default='outputs/output_transfer_learning',
                    help='the output foler that contains images of the learning process')
parser.add_argument('--load_model', type=str, default='weights/sepconv_weights',
                    help='the folder that contains the pretrained weights for us to do transfer learning on')
parser.add_argument('--save_weights', type=str, default='weights/transfer_learning_weights',
                    help='the folder to save the fine-tuned weights to')
parser.add_argument('--save_plots', type=str, default='images',
                    help='the folder to save loss and psnr plots to.')
parser.add_argument('--params_path', type=str, default='experiments/params1.json',
                    help='the path to the json file that contains the hyper-parameters')
parser.add_argument('--log_path', type=str, default='log_files/transfer_learning.log',
                    help='the path to transfer learning log file')

args = parser.parse_args()

# make the paths from args parser
project_dir = Path(__file__).parent.parent
train_dataset_dir = Path(project_dir/args.train_dir)
val_dataset_dir = Path(project_dir/args.val_dir)
out_dir = Path(project_dir/args.out_dir)
plots_dir = Path(project_dir/args.save_plots)
# weights path
features_weight_path = Path(project_dir/args.load_model/'features-l1.pytorch')
kernels_weight_path = Path(project_dir/args.load_model/'kernels-l1.pytorch')
output_weights_dir = Path(project_dir/args.save_weights)
# params path
params_path = Path(project_dir / args.params_path)
params = Param(params_path)
# log path
log_path = Path(project_dir / args.log_path)
# other params
kernel = params.kernel_size
epochs = params.epochs
batch_size = params.batch_size

# make the folders to save weights and outputs images, and also the images for loss & psnr plots
distance = train_dataset_dir.stem.split('_')[1]
now = datetime.now()
date_time = now.strftime("%m.%d.%Y_%H-%M-%S")
state_dict_dir = output_weights_dir / (date_time + '_distance' + distance)
if not state_dict_dir.exists():
    state_dict_dir.mkdir()  # save weights to this folder
out_dir_datetime = out_dir / (date_time + '_distance' + distance)
if not out_dir_datetime.exists():
    out_dir_datetime.mkdir()  # save output images to this folder
if not plots_dir.exists():
    plots_dir.mkdir()  # save the plots to this folder


def train_and_evaluate():
    """
    Train and Evaluate the model
    """
    # get the logger
    set_logger(log_path)
    logging.info('---Start Transfer Learning---')
    logging.info(f'Train data directory: {args.train_dir}')
    logging.info(f'Validation data directory: {args.val_dir}')
    logging.info(f'Output images directory: {args.out_dir}')
    logging.info(f'Log info is saved to: {args.log_path}')

    # load the model and the weights (for each part)
    model = to_cuda(SepConvNet(kernel))
    model.features.load_state_dict(torch.load(features_weight_path))
    model.subnet_kernel.load_state_dict(torch.load(kernels_weight_path))

    # get the dataset and data loader
    train_dataset = InterpolationDataset(train_dataset_dir, im_extension=params.image_extension)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = InterpolationDataset(val_dataset_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # freeze the features part of the model
    for param in model.features.parameters():
        param.requires_grad = False

    # fine-tune the subnet_kernels part
    optimizer = optim.Adam(params=model.subnet_kernel.parameters(), lr=params.learning_rate)
    criterion = nn.MSELoss()
    running_loss = 0
    train_losses, val_losses, average_psnrs = [], [], []

    # logging the hyperparameters
    logging.info(f'learning rate: {params.learning_rate}, '
                 f'kernel_size: {params.kernel_size}, '
                 f'epochs: {params.epochs}, '
                 f'batch_size: {params.batch_size}, '
                 f'val_after: {params.val_after} epochs')

    # training
    for epoch in range(epochs):
        logging.info(f'|Training|, Epoch {epoch+1}/{epochs}')
        model.train()
        steps = 0

        # prepare the output dir for each epoch
        out_epoch_dir = out_dir_datetime / ('epoch' + str(epoch).zfill(3))
        if not out_epoch_dir.exists():
            out_epoch_dir.mkdir()

        with tqdm(total=len(train_loader)) as t:  # use tqdm for progress bar
            # train
            for first_frame, gt_frame, sec_frame, gt_names in train_loader:
                steps += 1
                optimizer.zero_grad()
                frame_out = model(first_frame, sec_frame)
                loss = criterion(frame_out, gt_frame)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # evaluation
                if steps % params.val_after == 0:
                    val_loss = 0
                    avg_psnr = []
                    model.eval()
                    with torch.no_grad():
                        for val_first_frame, val_gt_frame, val_sec_frame, val_gt_names in val_loader:
                            val_frame_out = model(val_first_frame, val_sec_frame)
                            batch_loss = criterion(val_frame_out, val_gt_frame)
                            val_loss += batch_loss.item()

                            # calculate accuracy (peak signal to noise ratio)
                            psnr_err = psnr(val_gt_frame, val_frame_out)
                            avg_psnr.append(psnr_err)

                            # write the files into output folder
                            for index, name in enumerate(val_gt_names):
                                output_path = (out_epoch_dir / name).as_posix()
                                imwrite(val_frame_out[index], output_path)

                    logging.info(f"--|Evaluating|, Epoch {epoch + 1}/{epochs}.. "
                                 f"Step {steps}.. "
                                 f"Train loss: {running_loss / params.val_after :.5f}.. "
                                 f"Val loss: {val_loss / len(val_loader):.5f}.. "
                                 f"Average PSNR: {np.mean(avg_psnr):.3f}")

                    train_losses.append(running_loss / params.val_after)
                    val_losses.append(val_loss / len(val_loader))
                    average_psnrs.append(np.mean(avg_psnr))

                    running_loss = 0
                    model.train()

                t.update()

        # save the model state dict after each epoch
        state_dict_name = ('epoch' + str(epoch+1).zfill(3) + '-batch_size' + str(batch_size) + '.pytorch')
        state_dict_path = state_dict_dir / state_dict_name
        logging.info(f'Saving the model to {state_dict_dir.name}/{state_dict_name}...')

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'kernel_size': kernel,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'avg_psnr': average_psnrs}, state_dict_path)

    return train_losses, val_losses, average_psnrs


if __name__ == '__main__':
    train_losses, val_losses, average_psnrs = train_and_evaluate()

    logging.info(f'Saving the plots to {plots_dir}')
    # plot the losses and save to plot_dir folder
    plot_losses(train_losses, val_losses)
    losses_figure_name = date_time + '_losses' + '_epochs' + str(epochs) + '_distance' + \
                         distance + '_lr' + str(params.learning_rate) + '.png'
    losses_figure_path = Path(plots_dir / losses_figure_name)
    plt.savefig(losses_figure_path)

    # plot the psnr and save to plot_dir folder
    plt.figure()
    plt.plot(average_psnrs, label='average PSNR')
    plt.legend(frameon=False)
    psnr_figure_name = date_time + '_psnr' + '_epochs' + str(epochs) + '_distance' + \
                       distance + '_lr' + str(params.learning_rate) + '.png'
    psnr_figure_path = Path(plots_dir / psnr_figure_name)
    plt.savefig(psnr_figure_path)
