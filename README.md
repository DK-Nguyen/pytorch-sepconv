# Project: Applying Video Interpolation on Densely Sampled Light Field  
This project is forked from https://github.com/HyeongminLEE/pytorch-sepconv.git as the starting point.

## Summary
* [Description](#Description)
* [Dataset](#Dataset)
* [Performance](#Performance)
* [How to run](#How-to-run)
* [Project structure](#Project-structure)

### Description

### Dataset

### Performance
* Computer Spec:  
Intel core i7 4.20GHz (8 CPUs), 32Gb Ram.  
GPU: NVIDIA GeForce GTX 1080, Total memory: 24Gb, Display Memory (VRAM): 8Gb, Shared Memory: 16Gb.  
* To interpolate 1 image from 2 inputs 1280x720 - 1.19Mb images:  
Time: ~ 2 seconds.  
Memory:  0.11 Gb (0.088 Gb for the model which can be reused, 0.022 Gb for the object that contains tensors of the 2 input images and the methods to be called).
* To interpolate the whole Castle folder with distances 4:  
Time:  
Memory:  
* To interpolate the whole Castle folder with distances 8:  
Time:  
Memory:  
* To interpolate the whole Castle folder with distances 16:  
Time:   
Memory:  
* To interpolate the whole Castle folder with distances 32:  
Time:  
Memory:  


