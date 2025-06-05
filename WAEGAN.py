import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import save_image
import GPUtil

def conv5x5(in_channels, out_channels, stride=1, padding=2, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose': return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    else:                   return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=1, stride=1))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

class BottleConv(nn.Module):
    """
    Collapses or expands (w.r.t to channels) latent layers.  Composed of a single conv1x1 layer
    """
    def __init__(self, in_channels, out_channels,):
        super(BottleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv1x1(self.in_channels, self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, from_side):
        from_side = self.bn(F.relu(self.conv1(from_side)))
        return from_side

class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True, latent_channels=1):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        print ("Generating an DownConv layer with {0} input channels, {1} output channels, and pooling={2}".format(in_channels, out_channels, pooling))

        self.conv1 = conv5x5(self.in_channels,  self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.do = nn.Dropout()
        self.co = nn.Dropout2d()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Should we batch norm each time?  Or one time??
        x = self.do(self.bn(F.selu(self.conv1(x))))
        x = self.do(self.bn(F.selu(self.conv2(x))))
        x = self.co(x)
        if self.pooling: x = self.pool(x)
        return x

class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose', latent_channels=1):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        print ("Generating an UpConv layer with {0} input channels, {1} output channels, and mergemode={2}".format(in_channels, out_channels, merge_mode))

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)
        self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.do = nn.Dropout()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(F.selu(self.conv1(x)))
        x = self.bn(F.selu(self.conv2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self, up_mode='transpose', merge_mode='concat', latent_size=8):
        super(Encoder, self).__init__()

        self.latent_size = latent_size
        #depth_channels = [(3,16),(16,40),(40,60),(60,200),(200,250),(250,250),(250,100)]
        depth_channels = [(3,16),(16,40),(40,60),(60,150),(150,250),(250,100)]
        self.nchannel_in_out = depth_channels[-1][1]

        self.down_convs = []
        for layer in depth_channels:
            ins  = layer[0]
            outs = layer[1]
            pooling = True

            down_conv = DownConv(ins, outs, pooling=pooling, latent_channels=1)
            self.down_convs.append(down_conv)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.fc = nn.Linear(self.nchannel_in_out * self.latent_size * self.latent_size, 512)

    def forward(self,x):
        for i, module in enumerate(self.down_convs):
            x = module(x)
        x = x.view(-1, self.nchannel_in_out * self.latent_size * self.latent_size)
        x = F.relu(self.fc(x))
        return x


class Decoder(nn.Module):
    def __init__(self, up_mode='transpose', merge_mode='concat', latent_size=8):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        #depth_channels = [(3,16),(16,30),(30,64),(64,100),(100,200),(200,300),(300,100)]
        depth_channels = [(3,16),(16,30),(30,64),(64,100),(100,200),(200,100)]
        self.nchannel_in_out = depth_channels[-1][1]

        self.up_convs = []
        for layer in reversed(depth_channels):
            ins  = layer[1]
            outs = layer[0]
            pooling = True

            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode, latent_channels=1)
            self.up_convs.append(up_conv)

        self.up_convs = nn.ModuleList(self.up_convs)
        self.fc = nn.Linear(512, self.nchannel_in_out * self.latent_size * self.latent_size)

    def forward(self,x):
        x = F.relu(self.fc(x))
        x = x.view(-1, self.nchannel_in_out , self.latent_size , self.latent_size)
        for i, module in enumerate(self.up_convs):
            x = module(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dim_h = 512
        self.n_z = 256

        self.main = nn.Sequential(
            nn.Linear(512,             self.dim_h  * 3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.dim_h  * 3, self.dim_h  * 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.dim_h  * 2, self.dim_h // 2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(self.dim_h // 2, self.dim_h // 4),
            nn.ReLU(True),
            nn.Linear(self.dim_h // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x
