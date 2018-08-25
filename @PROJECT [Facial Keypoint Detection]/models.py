## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 16, 5, stride=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, 10, stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 64, 15, stride=(1, 1))
        #self.conv4 = nn.Conv2d(64, 128, 5, stride=(1, 1))
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        #self.fc1 = nn.Linear(4096, 2048)
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 136)
        self.drop = nn.Dropout2d(p=0.1)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x))) # (1, 224, 224) -> (16, 220, 220) -> (16, 110, 110)
        x = self.pool(F.relu(self.conv2(x))) # (16, 110, 110) -> (32, 51, 51) -> (32, 25, 25)
        x = self.pool(F.relu(self.conv3(x))) # (32, 25, 25) -> (64, 11, 11) -> (64, 5, 5)
        #x = self.pool(F.relu(self.conv4(x))) # (64, 8, 8) -> (128, 4, 4) -> (128, 2, 2)
        
        x = x.view(x.size(0), -1) # (64, 8, 8) -> 4096
        #x = self.drop(F.relu(self.fc1(x))) # 4096 -> 2048
        x = self.drop(F.relu(self.fc1(x))) # 1600 -> 1024
        x = self.drop(F.relu(self.fc2(x))) # 1024 -> 512
        x = self.drop(F.relu(self.fc3(x))) # 512 -> 256
        x = self.fc4(x) # 256 -> 136
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
