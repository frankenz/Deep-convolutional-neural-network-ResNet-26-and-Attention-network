#BuildingBlocks
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from typing import List, Mapping, Optional

from math import sqrt
import random
import numpy as np


class TinyExtractor (nn.Module):
    def __init__(self, channels_out):
        super (TinyExtractor, self).__init__()
        self.L = channels_out
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.main = nn.Sequential(
            #nnb.RBGtoHEres(),
            ConvBlock(32,   32,   3, 0),  # 128 -> 64
            ConvBlock(32,   64,   3, 0),  # 128 -> 64
            ConvBlock(64,   64,   3, 0, downsample=True, max2d=True),  # 64  -> 32
            ConvBlock(64,  128,   3, 0),  # 64  -> 32
            ConvBlock(128, 128,   3, 0),  # 64  -> 32
            ConvBlock(128, self.L, 3, 0, downsample=True, max2d=True),  # 64  -> 32
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.L, self.L),
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CrossEntropyWithProbs(nn.Module):
    """Calculate cross-entropy loss with smoothing"""
    def __init__(self,  classes: int, smoothing=0.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        """
        Parameters
        ----------
        classes: How many clases
        weight: An optional [num_classes] array of weights to multiply the loss by per class
        reduction: One of "none", "mean", "sum", indicating whether to return one loss per data
        smoothing: Max smoothing to be applied

        Returns
        -------
        torch.Tensor:The calculated loss
        """
        super(CrossEntropyWithProbs, self).__init__()
        self.smoothing = smoothing
        self.num_classes = classes
        self.weight = weight
        self.reduction = reduction
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = smooth_one_hot(target, self.num_classes, self.smoothing)
        return cross_entropy_with_probs(input, target, self.weight, self.reduction)

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method with random confidence in [1-smoothing to 1.0]
    """
    smoothing = smoothing

    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist

def cross_entropy_with_probs(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean") -> torch.Tensor:
    """Calculate cross-entropy loss when targets are probabilities (floats), not ints.
    PyTorch's F.cross_entropy() method requires integer labels; it does accept
    probabilistic labels. We can, however, simulate such functionality with a for loop,
    calculating the loss contributed by each class and accumulating the results.
    Libraries such as keras do not require this workaround, as methods like
    "categorical_crossentropy" accept float labels natively.
    Note that the method signature is intentionally very similar to F.cross_entropy()
    so that it can be used as a drop-in replacement when target labels are changed from
    from a 1D tensor of ints to a 2D tensor of probabilities.
    Parameters
    ----------
    input
        A [num_points, num_classes] tensor of logits
    target
        A [num_points, num_classes] tensor of probabilistic target labels
    weight
        An optional [num_classes] array of weights to multiply the loss by per class
    reduction
        One of "none", "mean", "sum", indicating whether to return one loss per data
        point, the mean loss, or the sum of losses
    Returns
    -------
    torch.Tensor
        The calculated loss
    Raises
    ------
    ValueError
        If an invalid reduction keyword is submitted
    """
    num_points, num_classes = input.shape
    # Note that t.new_zeros, t.new_full put tensor on same device as t
    cum_losses = input.new_zeros(num_points)
    for y in range(num_classes):
        target_temp = input.new_full((num_points,), y, dtype=torch.long)
        y_loss = F.cross_entropy(input, target_temp, reduction="none")
        if weight is not None:
            y_loss = y_loss * weight[y]
        cum_losses += target[:, y].float() * y_loss

    if reduction == "none":
        return cum_losses
    elif reduction == "mean":
        return cum_losses.mean()
    elif reduction == "sum":
        return cum_losses.sum()
    else:
        raise ValueError("Keyword 'reduction' must be one of ['none', 'mean', 'sum']")

class ZeroDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ZeroDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = 1.0-p
        self.b = torch.distributions.Bernoulli(self.p)

    def forward(self, input):
        if   self.training and input.is_cuda: return input * self.b.sample(input.shape).cuda()
        elif self.training:                   return input * self.b.sample(input.shape)
        else:                                 return input

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + str(self.p)  + ')'

class BasicResBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64):
        super(BasicResBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=True, dilation=1)
        self.relu  = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=True, dilation=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size + 2, kernel_size + 2)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = self.weight * self.multiplier
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size + 2, kernel_size + 2)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = self.weight * self.multiplier
        weight = (
              weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class RBGtoHEres(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_rgb_from_hed = nn.Parameter(torch.tensor([
        [ 1.8874,  0.2780, -1.5554],
        [-1.4174,  0.8393,  1.1682],
        [-0.1583, -0.4823,  1.6774]]).view(3,3,1,1),requires_grad=False)
    def forward(self, input):
        out =  input + 2
        out = -out.log10()
        out =  F.conv2d(out, self.w_rgb_from_hed)
        out = -torch.pow(10,-out) + 2
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class LinearNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel,kernel_size, padding,
        kernel_size2=None, padding2=None,
        downsample=False, fused=False, max2d=False, fast=False, gan=False
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.gan = gan

        self.conv1 = nn.Sequential(
        # This is always the same
            nn.Conv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.1),
        )

        # Fused down sample with fanning in
        if downsample and fused:
            self.conv2 = nn.Sequential(
                FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.1),
            )

        # Maxpool2D downsample
        elif downsample and max2d:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.1),
            )

        # Strided downsample followed by a Maxpool2D
        elif downsample and fast:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 2, padding=0, stride=2),
                nn.MaxPool2d(2),
                nn.LeakyReLU(0.1),
            )

        # strided downsample
        elif downsample:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 2, padding=0, stride=2),
                nn.LeakyReLU(0.1),
            )

        # Just do a convolution only
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.1),
            )
#        self.do = nn.Dropout2d(p=0.2)
        self.isnorm = nn.InstanceNorm2d(out_channel, affine=False, track_running_stats=False)

    def forward(self, input):
        out = self.conv1(input)
        # out = self.do(out)
        out = self.conv2(out)
        # out = self.isnorm(out)
        return out


#
# class ConvResBlock(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, padding,
#         kernel_size2=None, padding2=None,
#         downsample=False, fused=False, max2d=False, fast=False, gan=False
#     ):
#         super().__init__()
#
#
#         if in_channel != out_channel
#         downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out


class ConvToChannelOnly(nn.Module):
    def __init__(self, in_channel, out_channel, input_dim_size):
        super().__init__()

        # Just do a convolution only
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,  1, padding=0),
            nn.SELU(),
            nn.Conv2d(out_channel, out_channel, input_dim_size, padding=0),
            nn.SELU(),
        )

    def forward(self, input):
        out = self.conv(input)
        return out

class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=256,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample and fused:
                self.conv1 = nn.Sequential(
                    FusedUpsample(
                        in_channel, out_channel, kernel_size, padding=padding
                    ),
                    #Blur(out_channel),
                )

            elif upsample:
                self.conv1 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear'),
                    EqualConv2d(
                        in_channel, out_channel, kernel_size, padding=padding
                    ),
                    #Blur(out_channel),
                )

            else:
                self.conv1 = EqualConv2d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out
