# Code that NeuralNet

An attempt to code differnt SOTA deep learning models using Tensorflow and PyTorch. All the model architure diagrams used in the Readme were considered for reference while coding the models. The tf and torch implementations of all the models are present in Tensorflow and Pytorch folders respectively. 

<p align="center">
<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
</p>

General Dependencies for Running the code:
```
python = 3.8
tensorflow = 2.6
pytorch = 1.8
numpy = 1.21
```

## Alexnet

Designed by Alex Krizhevsky in collaboration with Ilya Sutskever and Geoffrey Hinton (ImageNet Classification with Deep Convolutional Neural Networks)

AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. The original paper's primary result was that the depth of the model was essential for its high performance, which was computationally expensive, but made feasible due to the utilization of graphics processing units (GPUs) during training.

<img src='Assets/AlexNet.png'/>

[More about Alexnet](https://paperswithcode.com/paper/imagenet-classification-with-deep)

## VGG-19

Proposed by K. Simonyan and A. Zisserman in the paper “Very Deep Convolutional Networks for Large-Scale Image Recognition”

VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer. VGG-19 is a convolutional neural network that is 19 layers deep.

<img src='Assets/VGG-19.png'/>

[More about VGG](https://paperswithcode.com/method/vgg)

## ResNet

Introduced by He et al. in Deep Residual Learning for Image Recognition

Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping. They stack residual blocks ontop of each other to form network: e.g. a ResNet-50 has fifty layers using these blocks.

<img src='Assets/ResNet.png'/>

[More about ResNet](https://paperswithcode.com/method/resnet)

## InceptionNetV3

Introduced by Szegedy et al. in Rethinking the Inception Architecture for Computer Vision

Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).

<img src="Assets/InceptionV3Architecture.png"/>

[More about InceptionV3](https://paperswithcode.com/method/inception-v3)
