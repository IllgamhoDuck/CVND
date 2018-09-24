[//]: # (Image References)

[image1]: ./images/key_pts_example.png "Facial Keypoint Detection"

# Facial Keypoint Detection

## Project Overview

In this project, you’ll combine your knowledge of computer vision techniques and deep learning architectures to build a facial keypoint detection system. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face; examples of these keypoints are displayed below.

![Facial Keypoint Detection][image1]

The project will be broken up into a few main parts in four Python notebooks, **only Notebooks 2 and 3 (and the `models.py` file) will be graded**:

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__Notebook 3__ : Facial Keypoint Detection Using Haar Cascades and your Trained CNN

__Notebook 4__ : Fun Filters and Keypoint Uses



## Project Instructions

All of the starting code and resources you'll need to compete this project are in this Github repository. Before you can get started coding, you'll have to make sure that you have all the libraries and dependencies required to support this project. If you have already created a `cv-nd` environment for [exercise code](https://github.com/udacity/CVND_Exercises), then you can use that environment! If not, instructions for creation and activation are below.

*Note that this project does not require the use of GPU, so this repo does not include instructions for GPU setup.*


### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data.
```
git clone https://github.com/udacity/P1_Facial_Keypoints.git
cd P1_Facial_Keypoints
```

2. Create (and activate) a new environment, named `cv-nd` with Python 3.6. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n cv-nd python=3.6
	source activate cv-nd
	```
	- __Windows__: 
	```
	conda create --name cv-nd python=3.6
	activate cv-nd
	```
	
	At this point your command line should look something like: `(cv-nd) <User>:P1_Facial_Keypoints <user>$`. The `(cv-nd)` indicates that your environment has been activated, and you can proceed with further package installations.

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

6. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```


### Data

All of the data you'll need to train a neural network is in the P1_Facial_Keypoints repo, in the subdirectory `data`. In this folder are training and tests set of image/keypoint data, and their respective csv files. This will be further explored in Notebook 1: Loading and Visualizing Data, and you're encouraged to look trough these folders on your own, too.


## Data Augmentation

A common strategy for training neural networks is to introduce randomness in the input data itself. For example, you can randomly rotate, mirror, scale, and/or crop your images during training. This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.

In this project, we have to be cautious of the kind of augmentation we choose to apply, since it will have a consequence to change the keypoints locations. We will only use random cropping to augment our train data which is implemented in`data_load`

## Transforms
We only augment our train dataset to be able to learn underlying patterns for achieving better results on the test dataset.

### Train dataset
With that in mind, here is how the pipeline to transform the train dataset looks like:
* Rescale the image to (250, 250) for the width and height
* Random crop the image to (227, 227)
* Normalize : convert color image to grayscale and normalize the color range to [0, 1] as well as scale keypoints around 0 with a range of [-1, 1]
* Convert to tensor

### Test dataset
 * Rescale the image to (227, 227) for the width and height
 *  Normalize : convert color image to grayscale and normalize the color range to [0, 1] as well as scale keypoints around 0 with a range of [-1, 1]
* Convert to tensor

## Model architecture
I first tried **Naimish** architecture from the paper 
`Facial Key Points Detection using Deep Convolutional Neural Network, N. Agarwal et al. (2016)` but it turns out that the model takes an input image of size (96, 96) which is fairly small compared to the size of our original images, so at the end we lose a lot of information by rescaling directly to this size.

I then turned towards one of the model that has given rise to the deep learning field, which is **AlexNet**. The *AlexNet* CNN architecture, `ImageNet Classification with Deep Convolutional Neural Networks, A. Krizhevsky et al. (2012)`, won the 2012 ImageNet ILSVRC challenge by a large margin: it achieved 17% top-5 error rate while the second best achieved only 26% ! 

However, I did change the activation functions and added some regularization techniques that helped improve the model.

| Layer               	| Details                                                                                          	|
|---------------------	|--------------------------------------------------------------------------------------------------	|
| Input               	| size : (227, 227, 1)                                                                             	|
| Conv 1              	| # filters : 96;  kernel size : (4 x 4);  stride : (4 x 4);  <br>padding : 0;   activation : ELU          	|
| Max Pooling         	| kernel size : (3 x 3);  stride : (2 x 2);  padding : 0 (VALID)                                     	|
| Dropout             	| probability : 0.2                                                                                	|
| Conv 2              	| # filters : 256;  kernel size : (5 x 5);  stride : (1 x 1);  <br>padding : 2 (SAME);   activation : ELU 	|
| Batch Normalization 	| # num_features: 256                                                                                  	|
| Max Pooling         	| kernel size : (3 x 3);  stride : (2 x 2);  padding : 0 (VALID)                                     	|
| Conv 3              	| # filters : 384;  kernel size : (3 x 3); stride : (1 x 1); <br>padding : 1 (SAME); activation : ELU   	|
| Batch Normalization 	| # num_features: 384                                                                                  	|
| Dropout             	| probability : 0.4                                                                                	|
| Conv 4              	| # filters : 384;  kernel size : (3 x 3);  stride : (1 x 1);  <br>padding : 1 (SAME);   activation : ELU  	|
| Batch Normalization 	| # filters : 384                                                                                  	|
| Dropout             	| probability : 0.4                                                                                	|
| Conv 5              	| # filters : 384;  kernel size : (3 x 3);  stride : (1 x 1);  <br>padding : 1 (SAME);   activation : ELU  	|
| Batch Normalization 	| # num_features: 384                                                                                  	|
| Dropout             	| probability : 0.4                                                                                	|
| Flatten             | (6 x 6 x 256) => 9216               |
| Fully Connected 1   | # neurons : 4096; activation : ELU   |
| Batch Normalization | # num_features: 4096                |
| Dropout             | probability : 0.6                   |
| Fully Connected 2   | # neurons : 4096; activation : ELU   |
| Batch Normalization | # num_features: 4096                |
| Dropout             | probability : 0.6                   |
| probability : 0.6   | # neurons : 136; activation : None |
| Output              | size : (136 x 1)                    |



LICENSE: This project is licensed under the terms of the MIT license.
