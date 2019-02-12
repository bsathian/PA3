#!/usr/bin/env python
# coding: utf-8

# In[ ]:


################################################################################
# CSE 253: Programming Assignment 3
# Winter 2019
# Code author: Jenny Hamer (+ modifications by Tejash Desai)
#
# Filename: baseline_cnn.py
#
# Description:
#
# This file contains the starter code for the baseline architecture you will use
# to get a little practice with PyTorch and compare the results of with your
# improved architecture.
#
# Be sure to fill in the code in the areas marked #TODO.
################################################################################


# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
from xray_dataloader import ChestXrayDataset, create_split_loaders

import matplotlib.pyplot as plt
import numpy as np
import os



class Arch1CNN(nn.Module):
    """
    conv1 -> conv2 -> maxpool -> conv3 -> conv4 -> conv5 -> maxpool -> fc1 -> fc2 -> fc3 (outputs)
    """

    def __init__(self):
        super(Arch1CNN, self).__init__()

        # conv1: 1 input channel, 12 output channels, [8x8] kernel size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=8)

        # Add batch-normalization to the outputs of conv1
        self.conv1_normed = nn.BatchNorm2d(12)

        # Initialized weights using the Xavier-Normal method
        torch_init.xavier_normal_(self.conv1.weight)

        #TODO: Fill in the remaining initializations replacing each '_' with
        # the necessary value based on the provided specs for each layer

        #TODO: conv2: X input channels, 10 output channels, [8x8] kernel, initialization: xavier
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=8)
        self.conv2_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv2.weight)
        #Maxpool
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)


        #TODO: conv3: X input channels, 12 output channels, [8x8] kernel, initialization: xavier
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=12, kernel_size=8)
        self.conv3_normed = nn.BatchNorm2d(12)
        torch_init.xavier_normal_(self.conv3.weight)
        #TODO: conv4: X input channels, 10 output channels, [6x6] kernel, initialization: xavier
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=10, kernel_size=6)
        self.conv4_normed = nn.BatchNorm2d(10)
        torch_init.xavier_normal_(self.conv4.weight)
        #TODO: conv5: X input channels, 8 output channels, [5x5] kernel, initialization: xavier
        self.conv5 = nn.Conv2d(in_channels=10, out_channels=8, kernel_size=5)
        self.conv5_normed = nn.BatchNorm2d(8)
        torch_init.xavier_normal_(self.conv5.weight)


        #TODO: Apply max-pooling with a [3x3] kernel using tiling (*NO SLIDING WINDOW*)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # Define 2 fully connected layers:
        #TODO: fc1
        self.fc1 = nn.Linear(in_features=160*160*8, out_features=512)
        self.fc1_normed = nn.BatchNorm1d(512)
        torch_init.xavier_normal_(self.fc1.weight)

        #TODO: fc2
        self.fc2 = nn.Linear(in_features=512, out_features=128)
        self.fc2_normed = nn.BatchNorm1d(128)
        torch_init.xavier_normal_(self.fc2.weight)

        #TODO: fc3
        self.fc3 = nn.Linear(in_features=128, out_features=14)
        torch_init.xavier_normal_(self.fc3.weight)

        #TODO: Output layer: what should out_features be?
        self.out_features = 14


    def forward(self, batch):
        """Pass the batch of images through each layer of the network, applying
        non-linearities after each layer.

        Note that this function *needs* to be called "forward" for PyTorch to
        automagically perform the forward pass.

        Params:
        -------
        - batch: (Tensor) An input batch of images

        Returns:
        --------
        - logits: (Variable) The output of the network
        """

        # Apply first convolution, followed by ReLU non-linearity;
        # use batch-normalization on its outputs
        batch = func.relu(self.conv1_normed(self.conv1(batch)))

        # Apply conv2 and conv3 similarly
        batch = func.relu(self.conv2_normed(self.conv2(batch)))
        batch = self.pool1(batch)

        batch = func.relu(self.conv3_normed(self.conv3(batch)))
        batch = func.relu(self.conv4_normed(self.conv4(batch)))
        batch = func.relu(self.conv5_normed(self.conv5(batch)))
        # Pass the output of conv3 to the pooling layer
        batch = self.pool2(batch)

        # Reshape the output of the conv3 to pass to fully-connected layer
        batch = batch.view(-1, self.num_flat_features(batch))

        # Connect the reshaped features of the pooled conv3 to fc1
        batch =  func.relu(self.fc1_normed(self.fc1(batch)))
        batch =  func.relu(self.fc2_normed(self.fc2(batch)))

        # Connect fc1 to fc2 - this layer is slightly different than the rest (why?)
        batch = self.fc3(batch)


        # Return the class predictions
        #TODO: apply an activition function to 'batch'
        batch = func.sigmoid(batch)
        return batch

    def num_flat_features(self, inputs):

        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1

        for s in size:
            num_features *= s

        return num_features

