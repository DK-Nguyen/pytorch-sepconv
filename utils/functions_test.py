"""
This Module is used to test if the functions and classes in the project work as expected
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from pathlib import Path
import PIL.Image as Image
import cv2

from model.layers.features_extraction import FeaturesExtraction
from model.layers.subnet_kernel import SubnetKernels
from model.model import SepConvNet
from utils.helpers import get_file_names, to_cuda, imshow
from utils.data_handler import InterpolationDataset

project_path = Path(__file__).parent.parent
# weights path
features_weight_path = Path(project_path/'weights'/'sepconv_weights/features-lf.pytorch')
kernels_weight_path = Path(project_path/'weights'/'sepconv_weights/kernels-lf.pytorch')
dslf_dataset_path = Path(project_path/'data'/'dslf_dataset')
# dataset path
castle_dataset_path = Path(project_path/'data'/'dslf'/'dslf')
holiday_dataset_path = Path(project_path/'data'/'dslf'/'dslf2')
seal_dataset_path = Path(project_path/'data'/'dslf'/'dslf4')
train_dataset_path = Path(project_path/'data'/'dslf'/'train')
val_dataset_path = Path(project_path/'data'/'dslf'/'val')


def reading_weights_check(features_path, kernels_path):
    """
    :param features_path: path to the features_extration weights
    :param kernels_path: path to the subnet_kernels weights
    """
    features_weight = torch.load(features_path)
    kernels_weight = torch.load(kernels_path)
    print(f"----Length of features weight: {len(list(features_weight))}")
    for param_name, weight in features_weight.items():
        print(param_name, weight.shape)
    assert len(list(features_weight)) == 62, "The feature extraction part " \
                                             "has less than 62 params"
    print(f"----Length of kernels weight: {len(list(kernels_weight))}----", )
    for param_name, weight in kernels_weight.items():
        print(param_name, weight.shape)
    assert len(list(kernels_weight)) == 32, "The separated kernels part" \
                                            "has less than 32 params"
    print('Passed reading split weights test')


def features_extraction_check():
    """
    After split the model and the pre-trained weights into features extraction and subnet kernels parts,
    use this function to test the features_extraction part
    """
    device = torch.device('cuda')
    start1 = time.time()
    feature_extraction = FeaturesExtraction().to(device)
    print('--- Loading weights for FeaturesExtraction ---')
    feature_extraction.load_state_dict(torch.load(features_weight_path))
    end1 = time.time()
    for layer_name, weights in feature_extraction.named_parameters():
        print(layer_name, weights.shape)
    print(f'Time to load the weights for FeaturesExtraction: {end1 - start1}s')
    print(f'Length of feature_extraction: {len(list(feature_extraction.parameters()))}')
    print(f'GPU Memory used: {torch.cuda.memory_allocated(device)/1000000:.2f} MB')

    print('--- Testing the features extraction part ---')
    start2 = time.time()
    # frame0 = to_cuda(torch.randn(1, 3, 1920, 1216))  # needs more than 8GB of GPU to run this
    # frame2 = to_cuda(torch.randn(1, 3, 1920, 1216))
    frame0 = to_cuda(torch.randn(1, 3, 1280, 736))
    frame2 = to_cuda(torch.randn(1, 3, 1280, 736))
    out = feature_extraction(frame0, frame2)
    end2 = time.time()
    assert out.shape[1] == 64
    assert out.shape[2] * 2 == frame0.shape[2]
    assert out.shape[3] * 2 == frame0.shape[3]
    print(f'Input shape: {frame0.shape}')
    print(f'Output shape: {out.shape}')
    print(f'Time to extract the features with 1 input pair: {end2 - start2}s')
    print(f'GPU Memory used: {torch.cuda.memory_allocated(device)/1000000:.2f} MB')
    print('Passed Features Extraction Test')


def subnet_kernel_check():
    """
    Test the subnet_kernel part of the split network
    """
    device = torch.device('cuda')
    start1 = time.time()
    subnet_kernel = SubnetKernels(subnet_kernel_size=51).to(device)
    print('--- Loading weights for SubnetKernel ---')
    subnet_kernel.load_state_dict(torch.load(kernels_weight_path))
    end1 = time.time()
    for layer_name, weights in subnet_kernel.named_parameters():
        print(layer_name, weights.shape)
    print(f'Length of subnet_kernel: {len(list(subnet_kernel.parameters()))}')
    print(f'Time to load the weights for SubnetKernel: {end1 - start1}s')
    print(f'GPU Memory used: {torch.cuda.memory_allocated(device)/1000000:.2f} MB')

    print('--- Test the SubnetKernel part ---')
    start2 = time.time()
    input_data = torch.randn(1, 64, 640, 368).cuda()
    out = subnet_kernel(input_data)
    end2 = time.time()
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {out[0].shape}')
    assert input_data.shape[2] * 2 == out[0].shape[2]
    assert input_data.shape[3] * 2 == out[0].shape[3]
    print(f'Time for 1 kernel to estimate the result: {end2 - start2}s')
    print(f'GPU Memory used: {torch.cuda.memory_allocated(device) / 1000000:.2f} MB')
    print('Passed subnet kernel test')


def model_check():
    """
    Check the whole model
    """
    device = torch.device('cuda')
    start1 = time.time()
    model = SepConvNet(subnet_kernel_size=51).to(device)
    print(f'--- Loading the weights for the SepConv model --- ')
    model.features.load_state_dict(torch.load(features_weight_path))
    model.subnet_kernel.load_state_dict(torch.load(kernels_weight_path))
    end1 = time.time()
    for layer_name, weights in model.named_parameters():
        print(layer_name, weights.shape)
    print(f'Time to load the model: {end1 - start1}s')
    print(f'GPU Memory used: {torch.cuda.memory_allocated(device)/1000000:.2f} MB')

    print('--- Test the Model ---')
    start2 = time.time()
    frame0 = to_cuda(torch.randn(1, 3, 1280, 720))
    frame2 = to_cuda(torch.randn(1, 3, 1280, 720))
    out = model(frame0, frame2)
    end2 = time.time()
    print(f'Input shape: {frame0.shape}')
    print(f'Output shape: {out.shape}')
    print(f'Time for the model to do 1 forward pass: {end2 - start2}s')
    print(f'GPU Memory used: {torch.cuda.memory_allocated(device) / 1000000:.2f} MB')
    print('Passed model test')


def pil_vs_cv2():
    """
    Test the performance of PIL and CV2 in reading images
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # testing time using PIL
    filenames = [f for f in castle_dataset_path.glob('*.png')]
    start_pill1 = time.time()
    pil1 = transform(Image.open(filenames[0]))
    end_pill1 = time.time()
    pil3 = [transform(Image.open(f)) for f in filenames[0:3]]
    end_pill3 = time.time()
    pill_all = [transform(Image.open(f)) for f in filenames]
    end_pil_all = time.time()
    print(f'Time for PIL to read and transform 1 image into tensor: {end_pill1 - start_pill1 :.2f}s')
    print(f'Time for PIL to read and transform 3 images into tensors: {end_pill3 - end_pill1 :.2f}s')
    print(f'Time for PIL to read and transform 193 images into tensors: {end_pil_all - end_pill3 :.2f}s')

    # testing time using OpenCV
    filenames = [f.absolute().as_posix() for f in castle_dataset_path.glob('*.png')]
    start_cv1 = time.time()
    cv1 = transform(cv2.imread(filenames[0]))
    end_cv1 = time.time()
    cv3 = [transform(cv2.imread(f)) for f in filenames[0:3]]
    end_cv3 = time.time()
    cv_all = [transform(cv2.imread(f)) for f in filenames]
    end_cv_all = time.time()
    print(f'Time for OpenCV to read and transform 1 image into tensor: {end_cv1 - start_cv1 :.2f}s')
    print(f'Time for OpenCV to read and transform 3 images into tensors: {end_cv3 - end_cv1 :.2f}s')
    print(f'Time for OpenCV to read and transform 193 images into tensors: {end_cv_all - end_cv3 :.2f}s')


def data_handler_check():
    val_dataset = InterpolationDataset(val_dataset_path)
    val_loader = DataLoader(val_dataset, batch_size=2)
    # for ff, gtf, sf in val_loader:
    a, b = val_dataset.get_path_lists()
    first_frame, gt_frame, sec_frame, gt_name = next(iter(val_loader))
    print(gt_name)
    print(first_frame.shape)
    print('Passed data_handler_check')


def get_files_name_check():
    get_file_names(castle_dataset_path, 4, print_file_names=True)
    print('Passed get files name for inference')


def debug_sepconv():
    pass


if __name__ == '__main__':
    # reading_weights_check(features_weight_path, kernels_weight_path)
    # features_extraction_check()
    # subnet_kernel_check()
    # model_check()
    # pil_vs_cv2()
    # get_files_name_check()
    # data_handler_check()

    # debug sepconv

    model = to_cuda(SepConvNet(51))
    model.features.load_state_dict(torch.load(features_weight_path))
    model.subnet_kernel.load_state_dict(torch.load(kernels_weight_path))
    frame0 = to_cuda(torch.randn(1, 3, 1280, 720))
    frame2 = to_cuda(torch.randn(1, 3, 1280, 720))
    frame_out = model(frame0, frame2)

