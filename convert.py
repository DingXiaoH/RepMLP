import argparse
import os
import torch
from repmlpnet import get_RepMLPNet_model

parser = argparse.ArgumentParser(description='RepMLPNet Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the source weights file')
parser.add_argument('save', metavar='SAVE', help='path to the target weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepMLPNet-B224')

def convert():
    args = parser.parse_args()
    model = get_RepMLPNet_model(args.arch, deploy=False)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        model.load_state_dict(ckpt)
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.load))

    model.locality_injection()

    torch.save(model.state_dict(), args.save)

if __name__ == '__main__':
    convert()