import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.elu = nn.ELU()
        self.downsample = downsample
        self.stride = stride
        
        self.shortcut = nn.Sequential(nn.Conv2d(inplanes, self.expansion*planes, kernel_size=1, stride=stride))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.elu(out)

        out = self.conv2(out)
        out = self.elu(out)

        out = self.conv3(out)
        
        out += self.shortcut(x)
        out = self.elu(out)

        return out

class DecoderBlock(nn.Module):

    def __init__(self, inplanes, planes, mid, stride=1):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid, planes, kernel_size=3, stride=1, padding=1)
        self.elu = nn.ELU()

    def forward(self, x, skip, udisp=None):
        x = self.upsample(x)
        x = self.elu(self.conv1(x))
        #concat
        if udisp is not None:
            x = torch.cat((x, skip, udisp), 1)
        else:
            x = torch.cat((x, skip), 1)
        x = self.elu(self.conv2(x))
        return x

class DispBlock(nn.Module):
    
    def __init__(self, inplanes, planes=2, kernel=3, stride=1):
        super(DispBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2, kernel_size=kernel, stride=stride, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        disp = 0.3 * self.tanh(self.conv1(x))
        udisp = self.upsample(disp)
        out = torch.cat((disp[:,0,:,:].unsqueeze(1) * x.shape[3], disp[:,1,:,:].unsqueeze(1) * x.shape[2]), 1)
        return out, disp, udisp

class MonodepthNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, do_stereo = 1):
        self.inplanes = 64
        super(MonodepthNet, self).__init__()
        if do_stereo:
            self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.up6 = DecoderBlock(2048, 512, 1536)
        self.up5 = DecoderBlock(512, 256, 768)
        self.up4 = DecoderBlock(256, 128, 384)
        self.get_disp4 = DispBlock(128)
        self.up3 = DecoderBlock(128, 64, 130)
        self.get_disp3 = DispBlock(64)
        self.up2 = DecoderBlock(64, 32, 98)
        self.get_disp2 = DispBlock(32)
        self.up1 = DecoderBlock(32, 16, 18)
        self.get_disp1 = DispBlock(16)
                

    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, 1))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes, 1))
        layers.append(block(self.inplanes, planes, 2))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        #encoder
        x = self.conv1(x) #2
        conv1 = self.elu(x)
        pool1 = self.maxpool(conv1) #4
        conv2 = self.layer1(pool1) #8
        conv3 = self.layer2(conv2) #16
        conv4 = self.layer3(conv3) #32
        conv5 = self.layer4(conv4) #64
        
        #skip
        skip1 = conv1
        skip2 = pool1
        skip3 = conv2
        skip4 = conv3
        skip5 = conv4
        
        #decoder
        upconv6 = self.up6(conv5, skip5)
        upconv5 = self.up5(upconv6, skip4)
        upconv4 = self.up4(upconv5, skip3)
        self.disp4_scale, self.disp4, udisp4 = self.get_disp4(upconv4)

        upconv3 = self.up3(upconv4, skip2, udisp4)
        self.disp3_scale, self.disp3, udisp3 = self.get_disp3(upconv3)

        upconv2 = self.up2(upconv3, skip1, udisp3)
        self.disp2_scale, self.disp2, udisp2 = self.get_disp2(upconv2)

        upconv1 = self.up1(upconv2, udisp2)
        self.disp1_scale, self.disp1, udisp1 = self.get_disp1(upconv1)
        
        return [self.disp1_scale, self.disp2_scale, self.disp3_scale, self.disp4_scale], [self.disp1, self.disp2, self.disp3, self.disp4]
        