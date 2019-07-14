import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
from pathlib import Path
import argparse
import math
from tqdm import tqdm
from matplotlib import pyplot as plt
from datetime import datetime

from model.model import SepConvNet
from utils.helpers import to_cuda, imwrite, plot_losses
from utils.data_handler import InterpolationDataset


parser = argparse.ArgumentParser(description='SepConv Pytorch')
# params
parser.add_argument('--train_dir', type=str, default='data/train_64')
parser.add_argument('--val_dir', type=str, default='data/val_64')
parser.add_argument('--out_dir', type=str, default='outputs/output_dslf')
parser.add_argument('--load_model', type=str, default='weights/sepconv_weights')
parser.add_argument('--weights_dir', type=str, default='weights/transfer_learning_weights')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1)

args = parser.parse_args()

# make the paths from args parser
project_dir = Path(__file__).parent.parent
train_dataset_dir = Path(project_dir/args.train_dir)
val_dataset_dir = Path(project_dir/args.val_dir)
out_dir = Path(project_dir/args.out_dir)
# weights path
features_weight_path = Path(project_dir/args.load_model/'features-lf.pytorch')
kernels_weight_path = Path(project_dir/args.load_model/'kernels-lf.pytorch')
output_weights_dir = Path(project_dir/args.weights_dir)
# other params
kernel = args.kernel
epochs = args.epochs
batch_size = args.batch_size


def train_and_evaluate():
    # load the model and the weights (for each part)
    model = to_cuda(SepConvNet(kernel))
    model.features.load_state_dict(torch.load(features_weight_path))
    model.subnet_kernel.load_state_dict(torch.load(kernels_weight_path))

    # get the dataset and data loader
    train_dataset = InterpolationDataset(train_dataset_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = InterpolationDataset(val_dataset_dir)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # freeze the features part of the model
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(params=model.subnet_kernel.parameters(), lr=0.001)  # fine-tune the subnet_kernels part
    criterion = nn.MSELoss()
    val_after = 50  # after training this number of batches, do evaluation
    running_loss = 0
    train_losses, val_losses, average_psnr = [], [], []

    # make the folder to save weights and outputs
    distance = train_dataset_dir.stem.split('_')[1]
    now = datetime.now()
    date_time = now.strftime("%m.%d.%Y_%H-%M-%S")
    state_dict_dir = output_weights_dir / (date_time + '_distance' + distance)
    if not state_dict_dir.exists():
        state_dict_dir.mkdir()
    out_dir_datetime = out_dir / (date_time + '_distance' + distance)
    if not out_dir_datetime.exists():
        out_dir_datetime.mkdir()

    # training
    for epoch in range(epochs):
        steps = 0
        # prepare the output dir for each epoch
        out_epoch_dir = out_dir_datetime / ('epoch' + str(epoch).zfill(3))
        if not out_epoch_dir.exists():
            out_epoch_dir.mkdir()

        with tqdm(total=len(train_loader)) as t:  # use tqdm for progress bar
            for first_frame, gt_frame, sec_frame, gt_names in train_loader:
                steps += 1
                optimizer.zero_grad()
                frame_out = model(first_frame, sec_frame)
                loss = criterion(frame_out, gt_frame)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # evaluation
                if steps % val_after == 0:
                    print(' |Evaluating| ', end=' ')
                    val_loss = 0
                    avg_psnr = 0
                    model.eval()
                    with torch.no_grad():
                        for val_first_frame, val_gt_frame, val_sec_frame, val_gt_names in val_loader:
                            val_frame_out = model(val_first_frame, val_sec_frame)
                            batch_loss = criterion(val_frame_out, val_gt_frame)
                            val_loss += batch_loss.item()

                            # calculate accuracy (peak signal to noise ratio)
                            psnr = -10 * math.log10(torch.mean((val_gt_frame - val_frame_out)
                                                               * (val_gt_frame - val_frame_out)).item())
                            avg_psnr += psnr
                            # write the files into output folder
                            for index, name in enumerate(val_gt_names):
                                output_path = (out_epoch_dir / name).as_posix()
                                imwrite(val_frame_out[index], output_path)

                    print(f"Epoch {epoch + 1}/{epochs}.. "
                          f"Step {steps}.. "
                          f"Train loss: {running_loss / val_after:.3f}.. "
                          f"Val loss: {val_loss / len(val_loader):.3f}.. "
                          f"Average PSNR: {avg_psnr / len(val_loader):.3f}")

                    train_losses.append(running_loss / val_after)
                    val_losses.append(val_loss / len(val_loader))
                    average_psnr.append(avg_psnr / len(val_loader))

                    running_loss = 0
                    model.train()

                t.update()

        # save the model state dict after each epoch
        state_dict_path = state_dict_dir / ('epoch' + str(epoch+1).zfill(3)
                                            + '-batch_size' + str(batch_size) + '.pytorch')
        print(f'Saving the model to {state_dict_path}...')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'kernel_size': kernel,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'avg_psnr': average_psnr}, state_dict_path)

    return train_losses, val_losses, average_psnr


if __name__ == '__main__':
    train_losses, val_losses, average_psnr = train_and_evaluate()
    plot_losses(train_losses, val_losses)
    plt.figure()
    plt.plot(average_psnr, label='average PSNR')
    plt.legend(frameon=False)
