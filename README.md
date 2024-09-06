# DLA
Repository containing the labs conducted for the Deep Learning Applications exam.
## Lab1-CNNs
### Required libraries to successfully run the code:
numpy, matplotlib, functools, torch, torchvision, tqdm, pandas, seaborn, sklearn, PIL

### Info
This laboratory aims to replicate, with simplified architectures, the results contained in the ResNet paper:

> [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016.

The goal is to show that increasing the depth (number of layers) of a network (whether linear or convolutional) does not result in higher accuracy for validation and test or in more reduction for training and validation loss.

The first section contains two exercises: 
* The first one involves training and evaluating an MLP architecture on the MNIST dataset while varying the depth.
  I set three different depth values (2, 5, 7), while keeping the rest of the architectures identical in terms of hyperparameters and seed, and then compared the test results. 
* The second exercise assesses the impact of increasing the depth of a CNN (defined with 3 main layers, each consisting of a variable number of blocks) trained on the CIFAR-10 dataset.
  Again, I considered three different depth values (8, 14, 16), keeping the hyperparameters and seed consistent across the models.

The second section aims to implement Grad-CAM according to the mathematical framework and guidelines from the referenced paper, with examples illustrating its application on a CNN model (defined during the previous exercises) and Cifar10 dataset. Feature maps from different layers were utilized to generate Grad-CAM, revealing the saliency points of each image at different depths within the network.

## Lab3-DRL
### Required libraries to successfully run the code:
numpy, matplotlib, gymnasium, torch, random, tqdm, collections, os, imageio, PIL

### Info
This laboratory is dedicated to implementing and executing some advanced Deep Reinforcement Learning algorithms. There are no studies on possible optimal hyperparameter values in the first two sections, as I empirically explored configurations of interest.

In the first section, the REINFORCE algorithm is implemented on the CartPole environment with a standard baseline. Since the policy is stochastic, 5 runs were performed for each configuration analyzed, and for each metric considered, the average curve and variance were examined.

The second section is based on the configurations identified in the first section, but this time the baseline is varied. The impact on the reference metrics is observed by either completely removing the baseline or using a neural network (Baseline Network).

Finally, the third section includes the training of some agents to solve the environments of CartPole and LunarLander using Deep Q Learning, where a DQN (simple Neural Network with 3 linear layers) approximates the Q-value function based on the state of the agent in the environment. The visual results of the agent's testing were saved as GIFs and included in the notebook.

## Lab4-OOD
### Required libraries to successfully run the code:
numpy, matplotlib, torch, torchvision, typing

### Info
