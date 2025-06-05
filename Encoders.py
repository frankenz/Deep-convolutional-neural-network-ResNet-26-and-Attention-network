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


class SMOTELayer(nn.Module):
    def __init__(self):
        super (SMOTELayer, self).__init__()
        self.epsilon = 0.005
        print ("Init SMOTELayer")
    def forward(self, x):
        pertubation = self.epsilon * (torch.randn(x.shape).cuda())
        #print ("Jostling with shape = ", pertubation.shape)
        #print (pertubation)
        return x + pertubation

class ClusterLayer(nn.Module):
    def __init__(self, nclus):
        super (ClusterLayer, self).__init__()
        self.n_clusters = nclus
        self.cluster_weights = nn.Parameter(torch.Tensor(nclus, 16*8))
        init.xavier_normal_(self.cluster_weights)
        print ("Init ClusterLayer with shape " + str(self.cluster_weights.shape))
    def forward(self, x):
        bn = x.size(0)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1) - self.cluster_weights
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        cl = x.argmin(dim=1).long().view(-1)
        x = x[torch.arange(x.size(0)), cl]
        interia_loss = torch.sum(x)
        xe_loss = torch.sum(torch.tensor([(1.0 if i==j else -1.0)*torch.dot(self.cluster_weights[i], self.cluster_weights[j]) for i in range(self.n_clusters) for j in range(self.n_clusters)]))

        return interia_loss/bn, xe_loss/self.n_clusters, cl

def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


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
        self.latent_channels = latent_channels
        self.pooling = pooling

        print ("Generating an DownConv layer with {0} input channels, {1} output channels, and pooling={2}".format(in_channels, out_channels, pooling))

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.do = nn.Dropout()
        self.co = nn.Dropout2d()

        self.bottle_conv1_in  = conv1x1(self.out_channels, self.latent_channels)
        self.bn_in  = nn.BatchNorm2d(self.latent_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Should we batch norm each time?  Or one time??
        x = self.do(self.bn(F.relu(self.conv1(x))))
        x = self.do(self.bn(F.relu(self.conv2(x))))
        x = self.co(x)
        from_down = x
        if self.pooling: x = self.pool(x)
        from_down = self.bn_in (F.relu(self.bottle_conv1_in (from_down)))

        return x, from_down


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
        self.latent_channels = latent_channels

        print ("Generating an UpConv layer with {0} input channels, {1} output channels, and mergemode={2}".format(in_channels, out_channels, merge_mode))

        self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

        if self.merge_mode == 'concat':
            # yall getting doulbe the channels from the concat channel
            self.conv1 = conv3x3(2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same since we add or skip
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.do = nn.Dropout()

        self.bottle_conv1_out = conv1x1(self.latent_channels, self.out_channels)
        self.bn_out = nn.BatchNorm2d(self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
            Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_down = self.bn_out(F.relu(self.bottle_conv1_out(from_down)))
        from_up = self.upconv(from_up)

        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        elif self.merge_mode == 'skip':
            x = from_up
        else:
            x = from_up + from_down

        x = self.bn(F.relu(self.conv1(x)))
        x = self.bn(F.relu(self.conv2(x)))
        return x

class LatentUNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, in_channels=3, out_channels=3, depth=5,
                 start_filts=16, latent_channels=10, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            out_channels: int, number of channels in the output tensor.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
            latent_channels: number of channels to push through the latent layer (lower means higher bottleneck)
        """
        super(LatentUNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add', 'skip'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_filts = start_filts
        self.depth = depth
        self.ndim_latent = latent_channels

        self.down_convs = []
        self.up_convs = []
        self.layermeta = []

        self.concat_layer = -1
        self.smote = SMOTELayer()

        self.fcl = nn.Linear(1024*8*8, 1024)

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins  = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv     = DownConv(ins, outs, pooling=pooling, latent_channels=self.ndim_latent)
            self.down_convs.append(down_conv)

	    # create bottle neck pathway
        self.bottle_neck_in    = BottleConv(outs,              self.ndim_latent)
        self.bottle_neck_out   = BottleConv(self.ndim_latent,  outs)

        # create the decoder pathway and add to a list - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2

            # HERE
            if i==self.concat_layer: merge_mode='concat'
            else: merge_mode='skip'

            up_conv = UpConv(ins, outs, up_mode=up_mode, merge_mode=merge_mode, latent_channels=self.ndim_latent)
            self.up_convs.append(up_conv)
            self.layermeta.append(merge_mode)

        self.conv_final = conv1x1(outs, self.out_channels)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs   = nn.ModuleList(self.up_convs)

        self.reset_params()
        self.first_time = False

    def UpdateClusterCenters(self, new_centers):
        self.clustercenters = new_centers
    def GetClusterCenters(self):
        return self.clustercenters

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias, mean=0, std=1)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, pertubation=False, early_stop=False):

        encoder_outs = 0
        if self.first_time: print (" +++ Input shape:" + str(x.shape))

        if self.first_time: print (" +++ Building Encoding Pathway +++ ")
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            if self.first_time: print ("Layer {0} :".format(i) + str(x.shape))
            x, before_pool = module(x)
            if i==(self.depth - self.concat_layer - 2): encoder_outs = before_pool

        if self.first_time: print ("+++ Bottlenecking Deepest Latent Layer +++ ")
        if self.first_time: print (x.shape)
        #latent_rep = self.bottle_neck_in(x)

        last_layer_flat = x.view(-1,  1024 * 8 * 8)

        latent_rep_flat = F.relu(self.fcl(last_layer_flat))

        latent_rep = latent_rep_flat.view(-1,  16,  8,  8)

        # Forget upconvolution, only return latent and sup rep
        if early_stop: return x, latent_rep_flat, encoder_outs

        if self.first_time: print ("+++ Rebuilding Decoder Inputs +++ ")
        # latent_rep  = self.smote(latent_rep)
        decoder_ins = self.smote(encoder_outs)

        if self.first_time: print (latent_rep.shape)
        x = self.bottle_neck_out(latent_rep)
        #x = self.rebuild(latent_rep)
        if self.first_time: print (x.shape)

        if self.first_time: print (" +++ Building Decoding Pathway +++ ")
        for i, module in enumerate(self.up_convs):
            if self.first_time: print ("Layer {0}, {1}, {2}:".format(i, self.layermeta[i], x.shape))
            x = module(decoder_ins, x)


        x = self.conv_final(x)
        if self.first_time: print (" +++ Output shape:")
        if self.first_time: print (x.shape)
        self.first_time = False

        return x, latent_rep_flat, encoder_outs
