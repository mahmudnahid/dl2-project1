# AASD4015 - Advanced Mathematical Concepts for Deep Learning

## Project: Analyze Residual Blocks & Upsampling Blocks for Enhanced Deep Residual Networks


### Team Members:
- Khandaker Nahid Mahmud (101427435)
- Siddhant Gite (101359755)

# Problem Statement: 

Image super-resolution (SR), particularly single image super-resolution (SISR) aims to reconstruct a high-resolution image from a single low-resolution image. Recent research on super-resolution has progressed with the development of deep convolutional neural networks (DCNN). In particular, residual learning techniques exhibit improved performance.

In the paper [Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)](https://arxiv.org/abs/1707.02921) Bee Lim et. al proposed the EDSR architecture, which is based on the SRResNet architecture.

In this project we implement the base single-scale model proposed in the paper and try to study the effect of Residual blocks and Upsampling blocks on image quality and training time. We analyzed the performance by changing the following parameters:

* Number of Residual blocks           
* Types of Upsampling blocks - Sub-Pixel CNN, Conv2DTranspose & UpSampling2D

# Introduction

The EDSR architecture is based on the SRResNet architecture and consists of multiple residual blocks. It uses constant  scaling layers instead of batch normalization layers to produce consistent results.

The residual block design of EDSR also differs from that of ResNet. Batch normalization layers have been removed (together with the final ReLU activation), since batch normalization layers normalize the features, they hurt output value range flexibility. It is thus better to remove them. Further, it also helps to reduce the amount of GPU RAM required by the model, since the batch normalization layers consume the same amount of memory as the preceding convolutional layers.

Reference: https://sh-tsang.medium.com/review-edsr-mdsr-enhanced-deep-residual-networks-for-single-image-super-resolution-super-4364f3b7f86f

**Modified Residual Blocks**

<img src="https://miro.medium.com/max/1050/1*EPviXGqlGWotVtV2gqVvNg.png" width="500" />


Apart from the residual blocks, another component of the architecture are the upsampling blocks. There are several strategies for upsampling and we explored the following:
1. Sub-Pixel CNN 
2. Conv2DTranspose 
3. UpSampling2D

We learned the concepts from the following articles and videos: 

https://www.analyticsvidhya.com/blog/2021/05/deep-learning-for-image-super-resolution/                    
http://krasserm.github.io/2019/09/04/super-resolution/                     
https://www.youtube.com/watch?v=fMwti6zFcYY&ab_channel=DigitalSreeni                  
https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d          


#### 1. Sub-Pixel CNN:
Given an input of size H×W×C and an upsampling factor s, the sub-pixel convolution layer first creates a representation of size H×W×s<sup>2</sup>C via a convolution operation and then reshapes it to sH×sW×C, completing the upsampling operation. The result is an output spatially scaled by factor s.

<img src="https://editor.analyticsvidhya.com/uploads/15959upsampling%20(3).JPG" width="500" />


#### 2. Transposed convolution:
Transposed convolution layer, tries to perform transformation opposite a normal convolution, i.e. predicting the possible  input based on feature maps sized like convolution output. Specifically, it increases the image resolution by expanding the image by inserting zeros and performing convolution.

<img src="https://editor.analyticsvidhya.com/uploads/55446upsampling%20(2).JPG" width="500" />


#### 3. Upsampling2D:
UpSampling2D is just a simple scaling up of the image by using nearest neighbour or bilinear upsampling. Advantage is it's cheap

<img src="https://i.imgur.com/UhwUfj0.png" width="500" />


The performance of the models are measured by the quality of generated images. To quantify reconstruction quality for images, Peak signal-to-noise ratio (PSNR) is measured and compared. 


# Dataset

### DIV2K:

We used the same dataset used in the paper - DIV2K. This is a popular single-image super-resolution dataset which contains 1,000 images with different scenes and is splitted to 800 for training, 100 for validation and 100 for testing. It was collected for NTIRE2017 and NTIRE2018 Super-Resolution Challenges in order to encourage research on image super-resolution with more realistic degradation. This dataset contains low resolution images with different types of degradations. Apart from the standard bicubic downsampling, several types of degradations are considered in synthesizing low resolution images for different tracks of the challenges. 


The dataset is available in tensorflow datasets: https://www.tensorflow.org/datasets/catalog/div2k


# Summary of Findings

INSERT TABLE


# Reference Notebook:

The EDSR notebook from Keras website is taken as reference to implement the base model. Then we experminted with various combinations of Rasidual Blocks and Upsampling Blocks. 

https://keras.io/examples/vision/edsr/

