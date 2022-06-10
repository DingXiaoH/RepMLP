# --------------------------------------------------------
# RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality (https://arxiv.org/abs/2112.11081)
# CVPR 2022
# Github source: https://github.com/DingXiaoH/RepMLP
# Licensed under The MIT License [see LICENSE for details]
# The training script is based on the code of Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# --------------------------------------------------------
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import load_weights, ProgressMeter, AverageMeter
from repmlpnet import get_RepMLPNet_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.utils import accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('data', metavar='DATA', help='path to dataset')
parser.add_argument('mode', metavar='MODE', default='train', choices=['train', 'deploy', 'check'], help='train, deploy, or check the equivalency?')
parser.add_argument('weights', metavar='WEIGHTS', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepMLPNet-B224')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 100) for test')
parser.add_argument('-r', '--resolution', default=224, type=int,
                    metavar='R',
                    help='resolution (default: 224) for test')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

def test():
    args = parser.parse_args()
    model = get_RepMLPNet_model(name=args.arch, deploy=args.mode == 'deploy')

    num_params = 0
    for k, v in model.state_dict().items():
        print(k, v.shape)
        num_params += v.nelement()
    print('total params: ', num_params)

    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        load_weights(model, args.weights)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.weights))

    if args.mode == 'check':    # Note this. In "check" mode, we load the trained weights and convert afterwards.
        model.locality_injection()

    if not torch.cuda.is_available():
        print('using CPU, this will be slow.')
        use_gpu = False
        criterion = nn.CrossEntropyLoss()
    else:
        model = model.cuda()
        use_gpu = True
        criterion = nn.CrossEntropyLoss().cuda()
        cudnn.benchmark = True

    t = []
    t.append(transforms.Resize(args.resolution))
    t.append(transforms.CenterCrop(args.resolution))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    trans = transforms.Compose(t)

    if os.path.exists('/home/dingxiaohan/ndp/imagenet.val.nori.list'):
        #   This is the data source on our machine. For debugging only. You will never need it.
        from noris_dataset import ImageNetNoriDataset
        val_dataset = ImageNetNoriDataset('/home/dingxiaohan/ndp/imagenet.val.nori.list', trans)
    else:
        #   Your ImageNet directory
        valdir = os.path.join(args.data, 'val')
        val_dataset = datasets.ImageFolder(valdir, trans)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, criterion, use_gpu)


def validate(val_loader, model, criterion, use_gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    test()