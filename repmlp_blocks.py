import torch
import torch.nn as nn
import torch.nn.functional as F
from repmlp import RepMLP, fuse_bn


#   RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition
#   Paper:  https://arxiv.org/abs/2105.01883
#   Code:   https://github.com/DingXiaoH/RepMLP

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = fuse_bn(self.conv, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv

class ConvBNReLU(ConvBN):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, deploy=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, deploy=deploy, nonlinear=nn.ReLU())


class RepMLPLightBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels,
                 H, W, h, w,
                 reparam_conv_k,
                 fc1_fc2_reduction,
                 fc3_groups,
                 deploy=False):
        super(RepMLPLightBlock, self).__init__()
        if in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, deploy=deploy)
        else:
            self.shortcut = nn.Identity()
        self.light_conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size=1, deploy=deploy)
        self.light_repmlp = RepMLP(in_channels=mid_channels, out_channels=mid_channels,
                                   H=H, W=W, h=h, w=w,
                                   reparam_conv_k=reparam_conv_k, fc1_fc2_reduction=fc1_fc2_reduction,
                                   fc3_groups=fc3_groups,
                                   deploy=deploy)
        self.repmlp_nonlinear = nn.ReLU()
        self.light_conv3 = ConvBN(mid_channels, out_channels, kernel_size=1, deploy=deploy)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.light_conv1(x)
        out = self.light_repmlp(out)
        out = self.repmlp_nonlinear(out)
        out = self.light_conv3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out