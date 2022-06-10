# --------------------------------------------------------
# RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (https://arxiv.org/abs/2112.11081)
# CVPR 2022
# Github source: https://github.com/DingXiaoH/RepMLP
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
    result.add_module('relu', nn.ReLU())
    return result

def fuse_bn(conv_or_fc, bn):
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)

    if len(t) == conv_or_fc.weight.size(0):
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return conv_or_fc.weight * repeated, (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(
            repeat_times, 0)


class GlobalPerceptron(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(GlobalPerceptron, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x

class RepMLPBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 h, w,
                 reparam_conv_k=None,
                 globalperceptron_reduce=4,
                 num_sharesets=1,
                 deploy=False):
        super().__init__()

        self.C = in_channels
        self.O = out_channels
        self.S = num_sharesets

        self.h, self.w = h, w

        self.deploy = deploy

        assert in_channels == out_channels
        self.gp = GlobalPerceptron(input_channels=in_channels, internal_neurons=in_channels // globalperceptron_reduce)

        self.fc3 = nn.Conv2d(self.h * self.w * num_sharesets, self.h * self.w * num_sharesets, 1, 1, 0, bias=deploy, groups=num_sharesets)
        if deploy:
            self.fc3_bn = nn.Identity()
        else:
            self.fc3_bn = nn.BatchNorm2d(num_sharesets)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = conv_bn(num_sharesets, num_sharesets, kernel_size=k, stride=1, padding=k//2, groups=num_sharesets)
                self.__setattr__('repconv{}'.format(k), conv_branch)


    def partition(self, x, h_parts, w_parts):
        x = x.reshape(-1, self.C, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x, h_parts, w_parts):
        fc_inputs = x.reshape(-1, self.S * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.S, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
        return out

    def forward(self, inputs):
        #   Global Perceptron
        global_vec = self.gp(inputs)

        origin_shape = inputs.size()
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(inputs, h_parts, w_parts)

        #   Channel Perceptron
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        #   Local Perceptron
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.S, self.h, self.w)
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = self.__getattr__('repconv{}'.format(k))
                conv_out += conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
            fc3_out += conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # N, O, h_parts, out_h, w_parts, out_w
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out


    def get_equivalent_fc3(self):
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = self.__getattr__('repconv{}'.format(largest_k))
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = self.__getattr__('repconv{}'.format(k))
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel += F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias += bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        self.deploy = True
        #   Locality Injection
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        #   Remove Local Perceptron
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                self.__delattr__('repconv{}'.format(k))
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(self.S * self.h * self.w, self.S * self.h * self.w, 1, 1, 0, bias=True, groups=self.S)
        self.fc3_bn = nn.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel, conv_bias):
        I = torch.eye(self.h * self.w).repeat(1, self.S).reshape(self.h * self.w, self.S, self.h, self.w).to(conv_kernel.device)
        fc_k = F.conv2d(I, conv_kernel, padding=(conv_kernel.size(2)//2,conv_kernel.size(3)//2), groups=self.S)
        fc_k = fc_k.reshape(self.h * self.w, self.S * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


#   The common FFN Block used in many Transformer and MLP models.
class FFNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0)
        self.ffn_fc2 = conv_bn(hidden_features, out_features, 1, 1, 0)
        self.act = act_layer()

    def forward(self, x):
        x = self.ffn_fc1(x)
        x = self.act(x)
        x = self.ffn_fc2(x)
        return x


class RepMLPNetUnit(nn.Module):

    def __init__(self, channels, h, w, reparam_conv_k, globalperceptron_reduce, ffn_expand=4,
                 num_sharesets=1, deploy=False):
        super().__init__()
        self.repmlp_block = RepMLPBlock(in_channels=channels, out_channels=channels, h=h, w=w,
                                        reparam_conv_k=reparam_conv_k, globalperceptron_reduce=globalperceptron_reduce,
                                        num_sharesets=num_sharesets, deploy=deploy)
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = x + self.repmlp_block(self.prebn1(x))  # TODO use droppath?
        z = y + self.ffn_block(self.prebn2(y))
        return z


class RepMLPNet(nn.Module):

    def __init__(self,
                 in_channels=3, num_class=1000,
                 patch_size=(4, 4),
                 num_blocks=(2,2,6,2), channels=(192,384,768,1536),
                 hs=(64,32,16,8), ws=(64,32,16,8),
                 sharesets_nums=(4,8,16,32),
                 reparam_conv_k=(3,),
                 globalperceptron_reduce=4, use_checkpoint=False,
                 deploy=False):
        super().__init__()
        num_stages = len(num_blocks)
        assert num_stages == len(channels)
        assert num_stages == len(hs)
        assert num_stages == len(ws)
        assert num_stages == len(sharesets_nums)

        self.conv_embedding = conv_bn_relu(in_channels, channels[0], kernel_size=patch_size, stride=patch_size, padding=0)

        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = [RepMLPNetUnit(channels=channels[stage_idx], h=hs[stage_idx], w=ws[stage_idx], reparam_conv_k=reparam_conv_k,
                                            globalperceptron_reduce=globalperceptron_reduce, ffn_expand=4, num_sharesets=sharesets_nums[stage_idx],
                                            deploy=deploy) for _ in range(num_blocks[stage_idx])]
            stages.append(nn.ModuleList(stage_blocks))
            if stage_idx < num_stages - 1:
                embeds.append(conv_bn_relu(in_channels=channels[stage_idx], out_channels=channels[stage_idx + 1], kernel_size=2, stride=2, padding=0))

        self.stages = nn.ModuleList(stages)
        self.embeds = nn.ModuleList(embeds)
        self.head_norm = nn.BatchNorm2d(channels[-1])
        self.head = nn.Linear(channels[-1], num_class)

        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        x = self.conv_embedding(x)
        for i, stage in enumerate(self.stages):
            for block in stage:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(embed, x)
                else:
                    x = embed(x)
        x = self.head_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def locality_injection(self):
        for m in self.modules():
            if hasattr(m, 'local_inject'):
                m.local_inject()

def create_RepMLPNet_T224(deploy=False):
    return RepMLPNet(channels=(64, 128, 256, 512), hs=(56,28,14,7), ws=(56,28,14,7),
                      num_blocks=(2,2,6,2), reparam_conv_k=(1, 3), sharesets_nums=(1,4,16,128),
                     deploy=deploy)
def create_RepMLPNet_T256(deploy=False):
    return RepMLPNet(channels=(64, 128, 256, 512), hs=(64,32,16,8), ws=(64,32,16,8),
                      num_blocks=(2,2,6,2), reparam_conv_k=(1, 3), sharesets_nums=(1,4,16,128),
                     deploy=deploy)
def create_RepMLPNet_B224(deploy=False):
    return RepMLPNet(channels=(96, 192, 384, 768), hs=(56,28,14,7), ws=(56,28,14,7),
                      num_blocks=(2,2,12,2), reparam_conv_k=(1, 3), sharesets_nums=(1,4,32,128),
                     deploy=deploy)
def create_RepMLPNet_B256(deploy=False):
    return RepMLPNet(channels=(96, 192, 384, 768), hs=(64,32,16,8), ws=(64,32,16,8),
                      num_blocks=(2,2,12,2), reparam_conv_k=(1, 3), sharesets_nums=(1,4,32,128),
                     deploy=deploy)
def create_RepMLPNet_D256(deploy=False):
    return RepMLPNet(channels=(80, 160, 320, 640), hs=(64,32,16,8), ws=(64,32,16,8),
                      num_blocks=(2,2,18,2), reparam_conv_k=(1, 3), sharesets_nums=(1,4,16,128),
                     deploy=deploy)
def create_RepMLPNet_L256(deploy=False):
    return RepMLPNet(channels=(96, 192, 384, 768), hs=(64,32,16,8), ws=(64,32,16,8),
                      num_blocks=(2,2,18,2), reparam_conv_k=(1, 3), sharesets_nums=(1,4,32,256),
                     deploy=deploy)

model_map = {
    'RepMLPNet-T256': create_RepMLPNet_T256,
    'RepMLPNet-T224': create_RepMLPNet_T224,
    'RepMLPNet-B224': create_RepMLPNet_B224,
    'RepMLPNet-B256': create_RepMLPNet_B256,
    'RepMLPNet-D256': create_RepMLPNet_D256,
    'RepMLPNet-L256': create_RepMLPNet_L256,
}

def get_RepMLPNet_model(name, deploy=False):
    if name not in model_map:
        raise ValueError('Not yet supported. You may add some code to create the model here.')
    model = model_map[name](deploy=deploy)
    return model


#   Verify the equivalency
if __name__ == '__main__':
    model = create_RepMLPNet_B224()
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    origin_y = model(x)

    model.locality_injection()

    print(model)
    new_y = model(x)
    print((new_y - origin_y).abs().sum())