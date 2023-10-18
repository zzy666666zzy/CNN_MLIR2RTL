#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 23:09:20 2023

@author: apple
"""

import torch.nn as nn

# Define a simple CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2)#(11,11)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 5, kernel_size=5, stride=1)#(3,3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(5, 7, kernel_size=5)#(3,3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(7, 10, kernel_size=3)#(1,1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = out.squeeze()
        return out