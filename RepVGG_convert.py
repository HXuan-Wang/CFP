import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn
from repvgg import get_RepVGG_func_by_name, repvgg_model_convert



def convert(args):
    repvgg_build_func = get_RepVGG_func_by_name(args.arch)
    train_model = repvgg_build_func(deploy=False, frozen=False,BBN=False)
    fc_features = train_model.linear.in_features
    train_model.linear = nn.Linear(fc_features, args.num_classes)

    train_model = train_model.cuda()

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        # checkpoint = torch.load(args.load)
        # train_model.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        elif 'model' in checkpoint:
            checkpoint = checkpoint['model']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        print(ckpt.keys())
        train_model.load_state_dict(ckpt,strict=False)

    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    if 'plus' in args.arch:
        train_model.switch_repvggplus_to_deploy()
        torch.save(train_model.state_dict(), args.save)
    else:
        repvgg_model_convert(train_model, save_path=args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RepVGG Conversion')
    parser.add_argument('--load', default='', help='path to the weights file')
    parser.add_argument('--save', default='', help='path to the weights file')
    parser.add_argument(
    '--num_classes',
    type = int,
    default = 10)

    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
    args = parser.parse_args()
    convert(args)