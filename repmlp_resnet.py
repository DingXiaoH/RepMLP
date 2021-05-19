import torch
import torch.nn as nn
import torch.nn.functional as F
from repmlp_blocks import ConvBNReLU, ConvBN, RepMLPLightBlock, RepMLPBottleneckBlock

#   RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition
#   Paper:  https://arxiv.org/abs/2105.01883
#   Code:   https://github.com/DingXiaoH/RepMLP

#   Original block of ResNet-50
class BaseBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride=1, deploy=False):
        super(BaseBlock, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, kernel_size=1, stride=stride, deploy=deploy)
        else:
            self.shortcut = nn.Identity()
        self.conv1 = ConvBNReLU(in_channels, mid_channels, kernel_size=1, deploy=deploy)
        self.conv2 = ConvBNReLU(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, deploy=deploy)
        self.conv3 = ConvBN(mid_channels, out_channels, kernel_size=1, deploy=deploy)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RepMLPResNet(nn.Module):
    def __init__(self, num_blocks, num_classes, block_type, img_H, img_W,
                 h, w, reparam_conv_k, fc1_fc2_reduction,
                 fc3_groups, deploy=False,
                 bottleneck_r=(2,4),        # r=2 for stage2 and r=4 for stage3
                 ):
        super(RepMLPResNet, self).__init__()

        #   base:   original ResNet-50
        #   light:  RepMLP Light Block (55% faster, comparable accuracy)
        #   bottleneck:     RepMLP Bottleneck Block (much higher accuracy, comparable speed)
        assert block_type in ['base', 'light', 'bottleneck']
        self.block_type = block_type
        self.deploy = deploy

        self.img_H = img_H
        self.img_W = img_W
        self.h = h
        self.w = w
        self.reparam_conv_k = reparam_conv_k
        self.fc1_fc2_reduction = fc1_fc2_reduction
        self.fc3_groups = fc3_groups
        self.bottleneck_r = bottleneck_r

        self.in_channels = 64
        channels = [256, 512, 1024, 2048]

        self.stage0 = nn.Sequential(
            ConvBNReLU(in_channels=3, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, deploy=deploy),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.stage1 = self._make_stage(channels[0], num_blocks[0], stride=1, total_downsample_ratio=4)
        self.stage2 = self._make_stage(channels[1], num_blocks[1], stride=2, total_downsample_ratio=8)
        self.stage3 = self._make_stage(channels[2], num_blocks[2], stride=2, total_downsample_ratio=16)
        self.stage4 = self._make_stage(channels[3], num_blocks[3], stride=2, total_downsample_ratio=32)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(channels[3], num_classes)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_stage(self, channels, num_blocks, stride, total_downsample_ratio):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for i, stride in enumerate(strides):
            if self.block_type == 'base' or stride == 2 or (total_downsample_ratio not in [8, 16]): #   Only use RepMLP in stage2 and stage3, as described in the paper
                cur_block = BaseBlock(in_channels=self.in_channels, mid_channels=channels // 4, out_channels=channels, stride=stride, deploy=self.deploy)
            elif self.block_type == 'light':
                cur_block = RepMLPLightBlock(in_channels=self.in_channels, mid_channels=channels // 8, out_channels=channels,
                                             H=self.img_H // total_downsample_ratio, W=self.img_W // total_downsample_ratio,
                                             h=self.h, w=self.w, reparam_conv_k=self.reparam_conv_k, fc1_fc2_reduction=self.fc1_fc2_reduction,
                                             fc3_groups=self.fc3_groups, deploy=self.deploy)
            elif self.block_type == 'bottleneck':
                cur_block = RepMLPBottleneckBlock(in_channels=self.in_channels, mid_channels=channels // 4,
                                             out_channels=channels,
                                             r = self.bottleneck_r[0] if total_downsample_ratio == 8 else self.bottleneck_r[1],
                                             H=self.img_H // total_downsample_ratio,
                                             W=self.img_W // total_downsample_ratio,
                                             h=self.h, w=self.w, reparam_conv_k=self.reparam_conv_k,
                                             fc1_fc2_reduction=self.fc1_fc2_reduction,
                                             fc3_groups=self.fc3_groups, deploy=self.deploy)
            else:
                raise ValueError('Not supported.')

            blocks.append(cur_block)
            self.in_channels = channels

        return nn.Sequential(*blocks)



def create_RepMLPRes50_Light_224(deploy):
    return RepMLPResNet(num_blocks=[3,4,6,3], num_classes=1000, block_type='light', img_H=224, img_W=224,
                        h=7, w=7, reparam_conv_k=(1,3,5), fc1_fc2_reduction=1, fc3_groups=4, deploy=deploy)
def create_RepMLPRes50_Bottleneck_224(deploy):
    return RepMLPResNet(num_blocks=[3,4,6,3], num_classes=1000, block_type='bottleneck', img_H=224, img_W=224,
                        h=7, w=7, reparam_conv_k=(1,3,5), fc1_fc2_reduction=1, fc3_groups=8, deploy=deploy, bottleneck_r=(2,4))