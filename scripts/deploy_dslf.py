import numpy as np
from math import log10
import math
import matplotlib.pyplot as plt
import os
import sys
from os import listdir
from os.path import join, isdir
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import cupy
import re
from torchvision.utils import save_image as imwrite
import itertools
import shutil
import sepconv
import argparse
import time

# commands for testing on small dataset
# python deploy_dslf.py --mode small --input my_input --output my_output

# exp commands for testing on dslf dataset
# python deploy_dslf.py --model l1 --distance 16

parser = argparse.ArgumentParser(description='DSLF Test')

# parameters
parser.add_argument('--mode', type=str, default='dslf')
parser.add_argument('--input', type=str, default='DSLF')
parser.add_argument('--output', type=str, default='DSLF')
parser.add_argument('--model', type=str, default='lf')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--distance', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=1)

# castleInput = os.path.join(os.getcwd(), 'DSLF', 'Castle')
# holidayInput = os.path.join(os.getcwd(), 'DSLF', 'Holiday')
# sealBallsInput = os.path.join(os.getcwd(), 'DSLF', 'Seal&Balls')

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class KernelEstimation(torch.nn.Module):
    def __init__(self, kernel_size):
        super(KernelEstimation, self).__init__()
        self.kernel_size = kernel_size

        def Basic(input_channel, output_channel):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Upsample(channel):
            return torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False)
            )

        def Subnet(ks):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=64, out_channels=ks, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=False),
                torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                torch.nn.Conv2d(in_channels=ks, out_channels=ks, kernel_size=3, stride=1, padding=1)
            )

        self.moduleConv1 = Basic(6, 32)
        self.modulePool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(32, 64)
        self.modulePool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(64, 128)
        self.modulePool3 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic(128, 256)
        self.modulePool4 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleConv5 = Basic(256, 512)
        self.modulePool5 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.moduleDeconv5 = Basic(512, 512)
        self.moduleUpsample5 = Upsample(512)

        self.moduleDeconv4 = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64)

        self.moduleVertical1 = Subnet(self.kernel_size)
        self.moduleVertical2 = Subnet(self.kernel_size)
        self.moduleHorizontal1 = Subnet(self.kernel_size)
        self.moduleHorizontal2 = Subnet(self.kernel_size)

        # use this line only when using the pretrained model to do inference. Comment it out when training
        self.load_state_dict(torch.load(modelPath))

    def forward(self, rfield0, rfield2):
        tensorJoin = torch.cat([rfield0, rfield2], 1)

        tensorConv1 = self.moduleConv1(tensorJoin)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)
        tensorPool4 = self.modulePool4(tensorConv4)

        tensorConv5 = self.moduleConv5(tensorPool4)
        tensorPool5 = self.modulePool5(tensorConv5)

        tensorDeconv5 = self.moduleDeconv5(tensorPool5)
        tensorUpsample5 = self.moduleUpsample5(tensorDeconv5)

        tensorCombine = tensorUpsample5 + tensorConv5

        tensorDeconv4 = self.moduleDeconv4(tensorCombine)
        tensorUpsample4 = self.moduleUpsample4(tensorDeconv4)

        tensorCombine = tensorUpsample4 + tensorConv4

        tensorDeconv3 = self.moduleDeconv3(tensorCombine)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)

        tensorCombine = tensorUpsample3 + tensorConv3

        tensorDeconv2 = self.moduleDeconv2(tensorCombine)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)

        tensorCombine = tensorUpsample2 + tensorConv2

        Vertical1 = self.moduleVertical1(tensorCombine)
        Vertical2 = self.moduleVertical2(tensorCombine)
        Horizontal1 = self.moduleHorizontal1(tensorCombine)
        Horizontal2 = self.moduleHorizontal2(tensorCombine)
        
        print(f'CUDA Memory Usage in Kernel Estimation: {torch.cuda.memory_allocated()/1000000000} Gb')
        return Vertical1, Horizontal1, Vertical2, Horizontal2
        

class SepConvNet(torch.nn.Module):
    def __init__(self, kernel_size):
        super(SepConvNet, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_pad = int(math.floor(kernel_size / 2.0))

        self.epoch = Variable(torch.tensor(0, requires_grad=False))
        self.get_kernel = KernelEstimation(self.kernel_size)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = torch.nn.MSELoss()

        self.modulePad = torch.nn.ReplicationPad2d([self.kernel_pad, self.kernel_pad, self.kernel_pad, self.kernel_pad])

    def forward(self, frame0, frame2):
        h0 = int(list(frame0.size())[2]) # height of the 1st input image
        w0 = int(list(frame0.size())[3]) # width of the 1st input image
        h2 = int(list(frame2.size())[2]) # height of the 2nd input image
        w2 = int(list(frame2.size())[3]) # width of the 2nd input image
        if h0 != h2 or w0 != w2:
            sys.exit('Frame sizes do not match')

        h_padded = False
        w_padded = False
        if h0 % 32 != 0:
            pad_h = 32 - (h0 % 32)
            frame0 = F.pad(frame0, (0, 0, 0, pad_h))
            frame2 = F.pad(frame2, (0, 0, 0, pad_h))
            h_padded = True

        if w0 % 32 != 0:
            pad_w = 32 - (w0 % 32)
            frame0 = F.pad(frame0, (0, pad_w, 0, 0))
            frame2 = F.pad(frame2, (0, pad_w, 0, 0))
            w_padded = True

        Vertical1, Horizontal1, Vertical2, Horizontal2 = self.get_kernel(frame0, frame2)
        tensorDot1 = sepconv.FunctionSepconv()(self.modulePad(frame0), Vertical1, Horizontal1)
        tensorDot2 = sepconv.FunctionSepconv()(self.modulePad(frame2), Vertical2, Horizontal2)
        frame1 = tensorDot1 + tensorDot2

        
        if h_padded:
            frame1 = frame1[:, :, 0:h0, :]
        if w_padded:
            frame1 = frame1[:, :, :, 0:w0]
        
        print(f'CUDA Memory Usage in SepConvNet: {torch.cuda.memory_allocated()/1000000000} Gb')
        
        return frame1
    
    
def getFileNames(inputDir, distance, printFileNames=False):
    '''
    Get the lists of corresponding file names we need to interpolate images
    params: inputDir - the path to the folder that contains ground truth images
            distance - (also called decimation) the distance between 2 ground truth 
                        images that we want to get the interpolated image of them
    output: fileNames - the map with keys being the rounds of the interpolating process,
                        the value of each key is a tuple containing the list of names  
                        of corressponding images for interpolating.
                        finally, the key 'final' contains the list of final names that we
                        need to keep for the interpolation result. This list will have the 
                        same number of images with the ground truth folder, so we can use
                        it to evaluate our method.
                        
                        An example: if distance = 4, then we have 2 rounds.
                        Round 1: Interpolating between 1 & 5 to get 3i, 5 & 9 to get 7i, ...
                                 The firstIms list is  [0001.png, 0005.png, 0009.png, ...]
                                 The secIms list is    [0005.png, 0009.png, 0013.png, ...]
                                 The outputIms list is [0003i.png, 0007i.png, 0011i.png, ...]
                        Round 2. Interpolating between 1 & 3i to get 2ii, 3i & 5 to get 4ii, ...
                                 The firstIms list is  [0001.png, 0003i.png, 0005.png, ...]
                                 The secIms list is    [0003i.png, 0005.png, 0007i.png, ...]
                                 The outputIms list is [0002ii.png, 0004ii.png, 0006ii.png, ...]
                        The final names: [0001.png, 0002ii.png, 0003i.png, 0004ii.png, 0005.png...]
                        
                        The return variable looks like this:
                        {1: ([001.png, 0005.png...], 
                            [0005.png, 0009.png...], 
                            [0003i.png, 0007i.png])
                         2: ([0001.png, 0003i.png,...]
                             [0003i.png, 0005.png,...]
                             [0002ii.png, 0004ii.png,...])
                         final: [0001.png, 0002ii.png, 0003i.png, 0004ii.png, 0005.png]
                        }
    '''
    
    numberOfRounds = math.log2(distance)
    assert distance >= 2 and distance <= 64, "distance must be in the range of [2, 64]"
    assert numberOfRounds % 1 == 0, "distance must be a power of 2 e.g. 2, 4, 8, 16..."
    gtFiles = [] # ground truth files
    for folders, subfolders, files in os.walk(inputDir):
        if '.ipynb_checkpoints' not in folders:
            gtFiles[:] = [f for f in files if not f.startswith("_")] # get all the names of the files in inputDir

    numberOfRounds = int(numberOfRounds)
    fileNames = {}
    for roundNumber in range(1, numberOfRounds + 1):
        fileNames[roundNumber] = ()
        if roundNumber == 1:
            firstIms = gtFiles[0::distance][:-1] 
            secIms = gtFiles[distance::distance]
            outputIms = gtFiles[int(distance/2)::distance]
            outputIms[:] = [(name.split('.')[0] + 'i' + '.' + name.split('.')[1]) for name in outputIms] # put 'i' into the names of interpolated files e.g. 0003.png -> 0003i.png
            assert len(firstIms) == len(secIms), "Lengths of first list and second list are different"
            assert len(firstIms) == len(outputIms), "Lengths of first list and output list are different"
            fileNames[roundNumber] += (firstIms, secIms, outputIms)
        else:
            # From round 2 onwards, the firstIms list is concatenated from the firstIms & outputIms of the previous round.
            # Similarly, the secIms list is concatenated from the secIms & outputIms of the previous round.
            firstIms = sorted(fileNames[roundNumber-1][0] + fileNames[roundNumber-1][2])
            secIms = sorted(fileNames[roundNumber-1][1] + fileNames[roundNumber-1][2])
            outputIms = gtFiles[int(distance/(2**roundNumber))::int(distance/(2**(roundNumber-1)))]
            outputIms[:] = [(name.split('.')[0] + 'i'*roundNumber + '.' + name.split('.')[1]) for name in outputIms]
            assert len(firstIms) == len(secIms), print("Lengths of first list and second list are different: {} vs {}".format(len(firstIms),len(secIms)))
            assert len(firstIms) == len(outputIms),  print("Lengths of first list and output list are different: {} vs {}".format(len(firstIms),len(outputIms)))
            fileNames[roundNumber] += (firstIms, secIms, outputIms)
    
    # final output is concatenated from all of the round's outputs, the first images and the last image of the first round.
    # the length must be 193
    fileNames['final'] = []
    for roundNumber in range(1, numberOfRounds + 1): # concatenating the outputs of all rounds
        fileNames['final'] += fileNames[roundNumber][2]
    fileNames['final'] += fileNames[1][0] # concatenating the first images of the first round
    fileNames['final'].append(fileNames[1][1][-1]) # concatenating the the last image.
    fileNames['final'] = sorted(fileNames['final'])
    assert len(fileNames['final']) == 193, print("Length of the final list is {}, not 193".format(len(fileNames['final'])))
    
    # print the fileNames 
    if printFileNames:
        for roundNum, fileNameLists in fileNames.items():
            if roundNum != 'final':
                print('Round Number {}, sizes of the lists: {}, {}, {}'.format(roundNum, len(fileNameLists[0]), len(fileNameLists[1]), len(fileNameLists[2])))
                print('First Images Names:', fileNameLists[0])
                print('Second Images Names:', fileNameLists[1])
                print('Interpolated Images Names:', fileNameLists[2])
                print()
            else:
                print('Final Names:', fileNameLists)
                
    return fileNames    

class smallTest:
    '''
    Used to test on small dataset (some images) when --mode is 'small'
    '''
    def __init__(self, input_dir):
        self.myTransform = transforms.Compose([transforms.ToTensor()])
        self.firstIms = []
        self.secIms = []
        
        for folder, subfolders, files in os.walk(input_dir):
            for file in sorted(files):
                filePath = os.path.join(os.path.abspath(folder), file)
                if ".ipynb_checkpoints" in filePath:
                    continue
#                 print(filePath)
                if file == "first.bmp":
                    self.firstIms.append(to_variable(self.myTransform(Image.open(filePath)).repeat(3, 1, 1).unsqueeze(0)))
                if file == "sec.bmp":
                    self.secIms.append(to_variable(self.myTransform(Image.open(filePath)).repeat(3, 1, 1).unsqueeze(0)))
        
    def test(self, model, ouput_dir, mode="one", idx=0):
        print("start testing, mode: {}".format(mode))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        frame_out = model(self.firstIms[idx], self.secIms[idx])
        imwrite(frame_out, os.path.join(ouput_dir, str(idx)+'.bmp'), range=(0, 1))
        print("testing done")
       
    def getFirstIms(self):
        return self.firstIms
    
    def getSecIms(self):
        return self.secIms

    
class OneImTest:
    '''
    Only used to test on one image at a time, doing this way we can save memory
    Used in the dslfTest function
    '''
    def __init__(self, firstImPath, secImPath):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.firstIm = to_variable(self.transform(Image.open(firstImPath)).unsqueeze(0))
        self.secIm = to_variable(self.transform(Image.open(secImPath)).unsqueeze(0))
        
    def test(self, model, outputPath):
        with torch.no_grad():
            frame_out = model(self.firstIm, self.secIm)
            imwrite(frame_out, os.path.join(outputPath), range=(0, 1))
            print("---------- Output: {} ----------".format(outputPath))


def dslfTest(dslfInput, fileNames, distance, model):
    '''
    run the model on the DSLF dataset - used when --mode is 'dslf'
    '''
    folderName = os.path.basename(os.path.normpath(dslfInput)) # get the name of the folder (Castle/Holiday/Seal&Balls)
    dslfOutputName = folderName + 'Interpolated' + str(distance) + args.model
    dslfOutput = os.path.join(os.getcwd(), 'DSLF', dslfOutputName)
    # make the output folder
    if not os.path.exists(dslfOutput):
        os.makedirs(dslfOutput)
        
    # copy the images in the firstIms into the output folder
    for name in fileNames[1][0]:
        shutil.copy(os.path.join(dslfInput, name), dslfOutput)
        
    # copy the last image of the secIms into the output folder
    shutil.copy(os.path.join(dslfInput, fileNames[1][1][-1]), dslfOutput)
    
    # for each round:
    #     if the key is not "final":
    #         for each image in the firstIms list:
    #             find the path to the corresponding second image and output image names
    #             apply the model on the first image and second image, output the image with corresponding name
    for key, value in fileNames.items(): # value is a tuple of lists: (firstIms, secIms, outputIms) for each round
        if key != 'final':
            print(f'round: {key}, dataset: {dslfOutput}, distance: {distance}')
            
            for idx, name in enumerate(value[0]):
                start1image = time.time()
                firstImPath = os.path.join(dslfOutput, name)
                secImPath = os.path.join(dslfOutput, value[1][idx])
                outputImPath = os.path.join(dslfOutput, value[2][idx])
                a = time.time()
                mytest = OneImTest(firstImPath, secImPath)
                b = time.time()
                mytest.test(model, outputImPath)
                c = time.time()
                print(f'Time to load the first and second image: {b-a}')
                print(f'Time to interpolate and write the image: {c-b}')
                print(f'Total time for interpolating 1 image: {time.time() - start1image}')
    
class DslfDataset2(Dataset):
    '''
    Try to apply the Dataset class from torch to do the interpolation to see if the performance is improved
    '''
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
        
        
def dslfTest2(dslfInput, fileNames, distance):
    folderName = os.path.basename(os.path.normpath(dslfInput)) # get the name of the folder (Castle/Holiday/Seal&Balls)
    dslfOutputName = folderName + 'Interpolated' + str(distance) + args.model
    dslfOutputPath = os.path.join(os.getcwd(), 'DSLF', dslfOutputName)
    # make the output folder
    if not os.path.exists(dslfOutputPath):
        os.makedirs(dslfOutputPath)
        
    # copy the images in the firstIms into the output folder (the images that we need to interpolate other images)
    for name in fileNames[1][0]:
        shutil.copy(os.path.join(dslfInput, name), dslfOutputPath)
        
    # the firstIms list does not contain the last image,
    # so we need to also copy the last image of the secIms into the output folder
    shutil.copy(os.path.join(dslfInput, fileNames[1][1][-1]), dslfOutputPath)
    
    model = SepConvNet(kernel_size=kernel_size).to('cuda')
    model.eval()
    start = time.time()
    
    # for each round:
    #     if the key is not "final":
    #         for each image in the firstIms list:
    #             find the path to the corresponding second image and output image names
    #             apply the model on the first image and second image, output the image with corresponding name
    for key, value in fileNames.items(): # value is a tuple of lists: (firstIms, secIms, outputIms) for each round
        testset = DslfDataset2(dslfOutputPath, value[0], value[1])
        testloader = DataLoader(testset, batch_size=1, shuffle=False)
        print(f'Memory in used: {torch.cuda.memory_allocated()/1000000000} Gb')
        if key != 'final':
            print(f'round: {key}, dataset: {dslfOutputPath}, distance: {distance}')
            counter = 0
            with torch.no_grad():
                for batch, (firstIm, secIm) in enumerate(testloader):
#                     print("Interpolating between {} and {}".format(firstIm, secIm))
                    frame_out = model(firstIm, secIm)
    
                    for i in range(frame_out.shape[0]):
                        outputImPath = os.path.join(dslfOutputPath, value[2][counter])
                        imwrite(frame_out[i:], outputImPath, range=(0, 1))
                        counter += 1
                        print("---------- Output: {} ----------".format(outputImPath))
                        
    end = time.time()
    print(f'Time for Interpolating: {end-start}')
        
        
        
if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    input_dir = os.path.join(os.getcwd(), args.input)
    output_dir = os.path.join(os.getcwd(), args.output)
    modelPath = os.path.join(os.getcwd(), 'network-' + args.model + '.pytorch')
    kernel_size = args.kernel
    distance = args.distance

    
    a = torch.cuda.memory_allocated()/1000000000
    model = SepConvNet(kernel_size=kernel_size).cuda().eval()
    b = torch.cuda.memory_allocated()/1000000000
    print(f'CUDA Memory for the Model: {b-a} Gb')
    
    if args.mode == 'small':
        start = time.time()
        mytest = smallTest(input_dir)
        c = torch.cuda.memory_allocated()/1000000000
        mytest.test(model, output_dir, "one", 0)
        print(f'Memory used by the object: {c-b} Gb')
        print('Deleting the object'); del mytest
        print('Deleting the model'); del model
        end = time.time()
        print(f'Time to test 1 image: {end-start} secs')
        print(f'Memory in used: {torch.cuda.memory_allocated()/1000000000} Gb')
        
    elif args.mode == 'dslf':
        castleInput = os.path.join(input_dir, 'Castle')
        holidayInput = os.path.join(input_dir, 'Holiday')
        sealBallsInput = os.path.join(input_dir, 'Seal&Balls')
        
        start = time.time()
        castleFileNames = getFileNames(castleInput, distance, printFileNames=False)
        dslfTest(castleInput, castleFileNames, distance, model)
        end = time.time()
        print(f'Total time for interpolating Castle with distance {distance}: {end-start} secs')
        
#         start = time.time()
#         holidayFileNames = getFileNames(holidayInput, distance, printFileNames=True)
#         dslfTest(holidayInput, holidayFileNames, distance, model)
#         end = time.time()
#         print(f'Interpolating Holiday with distance {distance} takes {end-start} secs')
        
#         start = time.time()
#         sealBallsFileNames = getFileNames(sealBallsInput, distance, printFileNames=True)
#         dslfTest(sealBallsInput, sealBallsFileNames, distance, model)
#         end = time.time()
#         print(f'Interpolating Seal&balls with distance {distance} takes {end-start} secs')



    # test dslfTest2 using the Dataset from torch to see if the performance is improved
#     castleInput = os.path.join(input_dir, 'Castle')
#     castleFileNames = getFileNames(castleInput, distance, printFileNames=True)
#     dslfTest2(castleInput, castleFileNames, distance)



