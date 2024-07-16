import logging
import numpy as np
from functools import partial
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Structure refers to paper (Re-)Imag(in)ing Price Trends
class CNN_5(nn.Module):
    def __init__(self, img_H=32, img_W=15, in_chans=3, kernel_size=(5, 3), padding=(2, 1), dilation=(3, 2)):
        self.in_chans = in_chans
        self.img_H = img_H
        self.img_W = img_W
        super(CNN_5, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64,
                      kernel_size=kernel_size, stride=(3, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=kernel_size, stride=(3, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.fc = nn.Sequential(nn.Linear(in_features=15360, out_features=2),
                                nn.Dropout(p=0.5), nn.Softmax(dim=1))  
        for layer in [self.conv1, self.conv2, self.fc]:
            self.weights_init_xavier(layer)

    def weights_init_xavier(self, layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        x = x.reshape(-1, self.in_chans, self.img_H, self.img_W) # Automatically match batch size
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CNN_20(nn.Module):
    def __init__(self, img_H=64, img_W=60, in_chans=3, kernel_size=(5, 3), padding=(2, 1), dilation=(1, 1)):
        super(CNN_20, self).__init__()
        self.in_chans = in_chans
        self.img_H = img_H
        self.img_W = img_W
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64,
                      kernel_size=kernel_size, stride=(2, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=kernel_size, stride=(2, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=kernel_size, stride=(2, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        # Adjusted in_features size based on the new dimensions
        self.fc = nn.Sequential(nn.Linear(in_features=15360, out_features=2),
                                nn.Dropout(p=0.1), nn.Softmax(dim=1))  
        for layer in [self.conv1, self.conv2, self.conv3, self.fc]:
            self.weights_init_xavier(layer)

    def weights_init_xavier(self, layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        x = x.reshape(-1, self.in_chans, self.img_H, self.img_W) # Automatically match batch size
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CNN_60(nn.Module):
    def __init__(self, img_H=96, img_W=180, in_chans=3, kernel_size=(5, 3), padding=(2, 1), dilation=(3, 2)):
        super(CNN_60, self).__init__()
        self.in_chans = in_chans
        self.img_H = img_H
        self.img_W = img_W
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64,
                      kernel_size=kernel_size, stride=(3, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=kernel_size, stride=(3, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=kernel_size, stride=(3, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=kernel_size, stride=(3, 1), padding=padding, dilation=dilation),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            nn.MaxPool2d((2, 1))
        )
        self.fc = nn.Sequential(nn.Linear(in_features=184320, out_features=2),
                                nn.Dropout(p=0.5), nn.Softmax(dim=1))  
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]:
            self.weights_init_xavier(layer)

    def weights_init_xavier(self, layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        x = x.reshape(-1, self.in_chans, self.img_H, self.img_W) # Automatically match batch size
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    # default is for 20-day pictures
    def __init__(self, img_H=64, img_W=60, in_chans=3, kernel_size=5):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=kernel_size, stride=1, padding=2), 
            nn.ReLU()
        )
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, stride=1, padding=2), 
            nn.ReLU()
        )
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64*int(img_H/4)*int(img_W/4), out_features=1000), 
            nn.Dropout(p=0.4), 
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=2), 
            nn.Softmax(dim=1)
        )
        for layer in [self.conv1, self.conv2, self.maxpool1, self.maxpool2, self.fc1, self.fc2]:
            self.weights_init_xavier(layer)

    def weights_init_xavier(self, layer):
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                init.zeros_(layer.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x) + x
        x = self.maxpool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x