# this module is used to run the demo on 2 images in the folder

import argparse
from pathlib import Path
from utils.helpers import to_cuda, imread, imwrite, psnr
from deploy_camera_rig import get_model
import time
import torch


parser = argparse.ArgumentParser(description='Run the demo')
parser.add_argument('--data_dir', type=str, default='data/demo',
                    help='directory that contains input images')
parser.add_argument('--output_dir', type=str, default='outputs/output_demo',
                    help='directory that contains output images')
parser.add_argument('--weight_path', type=str, default='weights/deploy_weights/network-l1',
                    help='the path to the pre-trained weights')
parser.add_argument('--image_extension', type=str, default='.png',
                    help='the image extension that you want to interpolate')

# prepare the path to the directories and the files that we need
args = parser.parse_args()
project_dir = Path(__file__).parent.parent
data_dir = Path(project_dir / args.data_dir)
output_dir = Path(project_dir / args.output_dir)
weight_path = Path(project_dir / args.weight_path)
first_im_path = Path(data_dir / "first.png")
sec_im_path = Path(data_dir / "sec.png")
out_im_path = Path(output_dir / "output.png")
gt_im_path = Path(data_dir / "gt.png")


if __name__ == "__main__":
    print('Load the model')
    model = get_model(weight_path)
    print(model)
    start = time.time()
    first_im = imread(first_im_path)
    # print(f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')
    sec_im = imread(sec_im_path)
    # print(f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')
    output_im = model(first_im, sec_im)
    # print(f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')
    imwrite(output_im, out_im_path, squeeze=True)
    end = time.time()
    print(f'time to interpolate 1 image of size 1280x720 is {end-start:.2f}s')
    print(f'CUDA Memory: {torch.cuda.memory_allocated()*1e-6:.2f} MB')
    print(f'psnr (peak signal to noise ratio): {psnr(imread(gt_im_path), output_im)}')
