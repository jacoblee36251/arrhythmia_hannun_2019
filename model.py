import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hannun2019(nn.Module):
    """CNN-based arrhythmia detection model used in Hannun et al 2019 paper (below)
    
    `Cardiologist-level arrhythmia detection and classification in ambulatory
     electrocardiograms using a deep neural network`

    Args:
        num_freqs (int): Number of frequency bands in the input signal
        vocab_size (int): The number of possible labels (including padding token)
        bias (bool): If True, enables biases on all CNN layers.
                     If False, only biases near beginning of network activated (just in case; to preserve info).
                     Reason for disabling biases: https://github.com/kuangliu/pytorch-cifar/issues/52
    """
    def __init__(self, num_freqs, vocab_size, bias=False):
        super(Hannun2019, self).__init__()
        
        strides = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        
        # initial input layers
        self.layers = [
            Conv1dSamePadding(num_freqs, 32, kernel_size=16, stride=1, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        ]
        
        # first resblock (modified, as per paper)
        self.layers.append(FirstBlock(in_channels=32, out_channels=64, kernel_size=16, stride=strides[0], bias=True))
        
        # rest of the 15 resblocks (16 total, including first above)
        k = 1
        for i in range(1, 16):
            if i % 4 == 0:
                k += 1
                self.layers.append(ResBlock(64*(k-1), 64*k, kernel_size=16, stride=strides[i], bias=bias))
            else:
                self.layers.append(ResBlock(64*k, 64*k, kernel_size=16, stride=strides[i], bias=bias))
            
        # Classifier layers
        self.layers.extend([
            nn.BatchNorm1d(64*k),
            nn.ReLU(inplace=True),
        ])
        
        self.layers = nn.Sequential(*self.layers)
        self.classifier = nn.Linear(64*k, vocab_size)
        
        self.epochs_trained = 0

    def forward(self, x, lens):
        x = self.layers(x)
        return self.classifier(x.permute(0,2,1)), lens // 2**8 - 1
    

class FirstBlock(nn.Module):
    """First resblock of the network, modified (as per their paper)

    Args:
        in_channels (int): number of channels in input
        out_channels (int): number of output filters
        kernel_size (int, optional): filter size (aka size of kernel). Defaults to 16.
        stride (int, optional): stride of kernel. Defaults to 1.
        bias (bool, optional): (See Hannun2019 docstring). Defaults to False.
    """
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, bias=False):
        super(FirstBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels, out_channels, kernel_size, stride=stride, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            Conv1dSamePadding(out_channels, out_channels, kernel_size, stride=1, bias=bias)
        )
        
        self.skip_connection = MaxPool1dSamePadding(kernel_size=stride)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            skip = F.pad(x, (0,0,0,self.out_channels-self.in_channels,0,0), "constant", 0)
            skip = self.skip_connection(skip)
        else:
            skip = self.skip_connection(x)
        x = self.layers(x)
        return x + skip


class ResBlock(nn.Module):
    """ResNet block based on He et al 2015 (Deep Residual Learning for Image Recognition)

    Args:
        in_channels (int): number of channels in input
        out_channels (int): number of output filters
        kernel_size (int, optional): filter size (aka size of kernel). Defaults to 16.
        stride (int, optional): stride of kernel. Defaults to 1.
        bias (bool, optional): (See Hannun2019 docstring). Defaults to False.
    """
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, bias=False):
        super(ResBlock, self).__init__()
    
        self.in_channels, self.out_channels = in_channels, out_channels
        
        self.layers = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            Conv1dSamePadding(in_channels, out_channels, kernel_size, stride=stride, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            Conv1dSamePadding(out_channels, out_channels, kernel_size, stride=1, bias=bias)
        )
        
        self.skip_connection = MaxPool1dSamePadding(kernel_size=stride)
        
    def forward(self, x):
        if self.in_channels != self.out_channels:
            skip = F.pad(x, (0,0,0,self.out_channels-self.in_channels,0,0), "constant", 0)
            skip = self.skip_connection(skip)
        else:
            skip = self.skip_connection(x)
        x = self.layers(x)
        return x + skip

    
class Conv1dSamePadding(nn.Conv1d):
    '''Wrapper around Conv1d. PyTorch version of padding=='same' from TensorFlow. Makes input_size == output_size.
    Only works when dilation == 1, groups == 1
    Code based on: https://github.com/pytPorch/pytorch/issues/3867
    '''
    def forward(self, x):
        batch_size, num_channels, length = x.shape
        if length % self.stride[0] == 0:
            out_length = length // self.stride[0]
        else:
            out_length = length // self.stride[0] + 1
        # pad = math.ceil((out_length * self.stride[0] + self.kernel_size[0] - length - self.stride[0]) / 2)
        pad = out_length * self.stride[0] + self.kernel_size[0] - length - self.stride[0]
        out = F.pad(x, (0,pad,0,0,0,0), "constant", 0)
        out = F.conv1d(input=out, weight=self.weight, stride=self.stride[0], bias=self.bias)
        # out = F.conv1d(input=x, weight=self.weight, stride=self.stride[0], bias=self.bias, padding=pad)
        if out.shape[-1] != length:
            out = out[:,:,:-1]
        return out


class MaxPool1dSamePadding(nn.MaxPool1d):
    '''Wrapper around MaxPool1d. PyTorch version of padding=='same' from TensorFlow. Makes input_size == output_size.
    Only works when dilation == 1, groups == 1
    Code based on: https://github.com/pytorch/pytorch/issues/3867
    '''
    def forward(self, x):
        batch_size, num_channels, length = x.shape
        if length % self.stride == 0:
            out_length = length // self.stride
        else:
            out_length = length // self.stride + 1
        # pad = math.ceil((out_length * self.stride + self.kernel_size - length - self.stride) / 2)
        pad = out_length * self.stride + self.kernel_size - length - self.stride
        out = F.pad(x, (0,pad,0,0,0,0), "constant", 0)
        out = F.max_pool1d(input=out, kernel_size=self.kernel_size, stride=self.stride)
        # out = F.max_pool1d(input=x, kernel_size=self.kernel_size, stride=self.stride, padding=pad)
        if out.shape[-1] != length:
            out = out[:,:,:-1]
        return out