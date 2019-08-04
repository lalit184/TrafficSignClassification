# keras-resnet
[![Build Status](https://travis-ci.org/raghakot/keras-resnet.svg?branch=master)](https://travis-ci.org/raghakot/keras-resnet)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/raghakot/keras-resnet/blob/master/LICENSE)

Residual networks implementation using Keras-1.0 functional API, that works with 
both theano/tensorflow backend and 'th'/'tf' image dim ordering.

### The original articles
 * [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385) (the 2015 ImageNet competition winner)
 * [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027)

### Residual blocks
The residual blocks are based on the new improved scheme proposed in [Identity Mappings in Deep Residual Networks](http://arxiv.org/abs/1603.05027) as shown in figure (b)

![Residual Block Scheme](images/residual_block.png?raw=true "Residual Block Scheme")

Both bottleneck and basic residual blocks are supported. To switch them, simply provide the block function [here](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L109)

### Code Walkthrough for RESNET50
The architecture is based on 50 layer sample (snippet from paper)

![Architecture Reference](images/architecture.png?raw=true "Architecture Reference")

There are two key aspects to note here

 1. conv2_1 has stride of (1, 1) while remaining conv layers has stride (2, 2) at the beginning of the block. This fact is expressed in the following [lines](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L63-L65).
 2. At the end of the first skip connection of a block, there is a disconnect in num_filters, width and height at the merge layer. This is addressed in [`_shortcut`](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L41) by using `conv 1X1` with an appropriate stride.
 For remaining cases, input is directly merged with residual block as identity.

### ResNetBuilder factory
- Use ResNetBuilder [build](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L135-L153) methods to build standard ResNet architectures with your own input shape. It will auto calculate paddings and final pooling layer filters for you.
- Use the generic [build](https://github.com/raghakot/keras-resnet/blob/master/resnet.py#L99) method to setup your own architecture.


### USAGE REQUIREMENTS
	
	1)python 2.7 or higgher
	2)tensorflow 
	3)keras
	4)python h5py
	5)PIL aka python image library


### USAGE DETAILS
	
	Training 
		1) Keep all files in /train directory with each images of same class inside same directory.
		i.e /tain/1  /train/2  /train/3 ....
		2) Specify the location of your /train folder in parameters.py ideally it should be within RESNET50 folder.
		3) change the appropriate parameters of your testing in parameters.py
		4) create a directory /check_point in RESNET50 directory and mention its location and model weights
		name in the form /loation/model_weights.h5 in save_dir  
		5)run train.py and the model will begin training 

### TODO
	
	1) train from a checkpoint 
	2) demo.py
	3) demo_ros.py
