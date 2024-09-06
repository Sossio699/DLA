# DLA
Repository containing the labs conducted for the Deep Learning Applications exam.
## Lab1
### Required libraries to successfully run the code:
numpy, matplotlib, functools, torch, torchvision, tqdm, pandas, seaborn, sklearn, PIL


This laboratory aims to replicate, with simplified architectures, the results contained in the ResNet paper:

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.

The goal is to show that increasing the depth (number of layers) of a network (whether linear or convolutional) does not result in higher accuracy for validation and test or in more reduction for training and validation loss.

The first section contains two exercises: 
* The first one involves training and evaluating an MLP architecture on the MNIST dataset while varying the depth.
  I set three different depth values (2, 5, 7), while keeping the rest of the architectures identical in terms of hyperparameters and seed, and then compared the test results. 
* The second exercise assesses the impact of increasing the depth of a CNN (defined with 3 main layers, each consisting of a variable number of blocks) trained on the CIFAR-10 dataset.
  Again, I considered three different depth values (8, 14, 16), keeping the hyperparameters and seed consistent across the models.

The second section aims to implement Grad-CAM according to the mathematical framework and guidelines from the referenced paper, with examples illustrating its application on a CNN model (defined during the previous exercises) and Cifar10 dataset. Feature maps from different layers were utilized to generate Grad-CAM, revealing the saliency points of each image at different depths within the network.

## Lab3
### Required libraries to successfully run the code:
numpy, matplotlib, gymnasium, torch, random, tqdm, collections, os, imageio, PIL

## Lab4
### Required libraries to successfully run the code:
numpy, matplotlib, torch, torchvision, typing
