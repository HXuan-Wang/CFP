
import os
import numpy as np
import time, datetime
import argparse
import copy
from thop import profile
from collections import OrderedDict
from st import *
import torch
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from tqdm import tqdm
import torchvision
from torchvision import datasets, transforms
import torchmetrics
from models.VGG16 import vgg_16_bn
from models.DDDN import DDDN
from models.repvgg import RepVGGA0
from models.repvgg import get_RepVGG_func_by_name
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from dataset import *

import utils.common as utils
from torch.utils.data.sampler import WeightedRandomSampler

parser = argparse.ArgumentParser("baseline training")

parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument(
    '--data_dir',
    type=str,
    default='./GC10-DET/images/images',
    help='path for dataset')
parser.add_argument(
    '--job_dir',
    type=str,
    default='./original_RepVGG_5',
    help='path for saving trained models')
parser.add_argument(
    '--dataset',
    type=str,
    default='GC10-DET',
    choices=('GC10-DET','NEU-CLS','ELPV','X-SDD'),
    help='cifar10 or imagenet')
parser.add_argument(
    '--batch_size',
    type=int,
    default=64,
    help='batch size')
parser.add_argument(
    '--epochs',
    type=int,
    default=300,
    help='num of training epochs')

parser.add_argument(
    '--input_size',
    type=int,
    default=224,
    help='size of input image')
parser.add_argument(
    '--num_classes',
    type=int,
    default=10,
    help='size of input image')
parser.add_argument('-lr', type=float, default=0.0005, help=' learning rate')  # 0.001
parser.add_argument(
    '--resume',
    default=False,
    help='whether continue training from the same directory')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./RepVGG-A0-train.pth',
    help='pretrain model path')
parser.add_argument(
    '--test_only',
    default=False,
    help='whether it is test mode')
parser.add_argument(
    '--test_model_dir',
    type=str,
    default='',
    help='test model path')
parser.add_argument(
    '--gpu',
    type=str,
    default='5',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
CLASSES = 10
print_freq = (256*50)//args.batch_size

if not os.path.isdir(args.job_dir):
    os.makedirs(args.job_dir)

utils.record_config(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))

#use for loading pretrain model
if len(args.gpu)>1:
    name_base='module.'
else:
    name_base=''


def main():

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)
    #loading model
    logger.info('==> Building model..')
    num_classes = args.num_classes
    repvgg_build_func = get_RepVGG_func_by_name('RepVGG-A0')
    model = repvgg_build_func(deploy=False)
    model.load_state_dict(torch.load(args.pretrain_dir))
    fc_features = model.linear.in_features
    model.linear = nn.Linear(fc_features, num_classes)
    model = model.cuda()

    logger.info(model)


    #calculate model size
    input_image_size=args.input_size
    input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
    flops, params = profile(model, inputs=(input_image,))
    logger.info('Params: %.2f M' % (params / 1e6))
    logger.info('Flops: %.2f G' % (flops / 1e9))

    input_size = args.input_size

    input_size = args.input_size

    train_transform = transforms.Compose([
        transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 ]),  # above is for: randomly selecting one for process
        # transforms.RandomRotation(degrees=2),  #
        transforms.RandomChoice([
            transforms.CenterCrop((input_size, input_size)),
            transforms.Resize((input_size, input_size)),
        ]),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if args.dataset == 'GC10-DET':
        train_data = MyDataset(fname=os.getcwd(), transform=train_transform, train=True)
        test_data = MyDataset(fname=os.getcwd(), transform=test_transform, train=False)

    elif args.dataset == 'X-SDD':
        train_data = Imbalanced_Dataset(fname=os.getcwd(), transform=train_transform, train=True)
        test_data = Imbalanced_Dataset(fname=os.getcwd(), transform=test_transform, train=False)
    elif args.dataset == 'NEU-CLS':
        train_data = MyDataset(fname=os.getcwd(), transform=train_transform, train=True)
        test_data = MyDataset(fname=os.getcwd(), transform=test_transform, train=False)

    elif args.dataset == 'ELPV':
        train_data = MyDataset(fname=os.getcwd() + "/elpv-dataset-master/utils", transform=train_transform, train=True)
        test_data = MyDataset(fname=os.getcwd() + "/elpv-dataset-master/utils", transform=test_transform, train=False)

    train_dataset = train_data
    test_dataset = test_data

    label_weight = []
    label_number = np.zeros(num_classes)
    for i, (data, label) in enumerate(train_dataset):
        label_weight.append(label)
        for j in range(num_classes):
            if label == j:
                label_number[j] += 1
    print(label_number)

    weights = 1. / torch.tensor(label_number, dtype=torch.float)

    samples_weights = weights[label_weight]

    sampler = WeightedRandomSampler(samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                              num_workers=4)
    val_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.test_only:
        if os.path.isfile(args.test_model_dir):
            logger.info('loading checkpoint {} ..........'.format(args.test_model_dir))
            checkpoint = torch.load(args.test_model_dir)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            test_acc = torchmetrics.Accuracy().cuda()
            test_recall = torchmetrics.Recall(average='macro', num_classes=num_classes).cuda()
            test_precision = torchmetrics.Precision(average='macro', num_classes=num_classes).cuda()
            test_F1score = torchmetrics.F1Score(average='macro', num_classes=num_classes).cuda()
            test_pbar = tqdm(enumerate(val_loader), total=len(val_loader))
            print(('\n' + '%15s' * 3) % ('Epoch', 'Epochs', 'test_accuracy'))
            with torch.no_grad():
                total_test_loss = 0
                for i, (test_iamges, test_labels) in test_pbar:
                    test_iamges = test_iamges.cuda()
                    test_labels = test_labels.cuda()
                    test_output = model(test_iamges).cuda()
                    _, predicted = torch.max(test_output.data, 1)
                    test_acc(predicted, test_labels)
                    test_recall(predicted, test_labels)
                    test_precision(predicted, test_labels)
                    test_F1score(predicted, test_labels)
                    m = ('%15s' * 2 + '%15.4g' * 1) % (0, args.epochs, 0)
                    test_pbar.set_description(m)
            print("accuracy, precision,recall,F1", test_acc.compute(), test_precision.compute(), test_recall.compute(),
                  test_F1score.compute())
        else:
            logger.info('please specify a checkpoint file')
        return

    if len(args.gpu) > 1:
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        model = nn.DataParallel(model, device_ids=device_id).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=0.001,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True, eta_min=1e-6)

    start_epoch = 0
    best_acc= 0

    # load the checkpoint if it exists

    checkpoint_dir = os.path.join(args.job_dir, 'model_best.pth.tar')


    if args.resume:
        logger.info('loading checkpoint {} ..........'.format(checkpoint_dir))
        checkpoint = torch.load(checkpoint_dir)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

        # deal with the single-multi GPU problem
        new_state_dict = OrderedDict()
        tmp_ckpt = checkpoint['state_dict']
        if len(args.gpu) > 1:
            for k, v in tmp_ckpt.items():
                new_state_dict['module.' + k.replace('module.', '')] = v
        else:
            for k, v in tmp_ckpt.items():
                new_state_dict[k.replace('module.', '')] = v

        model.load_state_dict(new_state_dict)
        logger.info("loaded checkpoint {} epoch = {}".format(checkpoint_dir, checkpoint['epoch']))


    # adjust the learning rate according to the checkpoint

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        print(('\n' + '%15s' * 4) % ('Epoch', 'Epochs', 'lr', 'train_loss'))
        train_correct = 0
        total_train_loss = 0
        for i, (images, labels) in pbar:
            images = images.cuda()
            labels = labels.long().cuda()

            output_s = model(images)
            loss_ce = criterion(output_s, labels).cuda()

            loss = loss_ce  # +kd_loss

            total_train_loss += loss
            _, predicted = torch.max(output_s.data, 1)
            train_correct += (predicted == labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s = ('%15s' * 2 + '%15.4g' * 2) % (
                epoch, args.epochs, optimizer.state_dict()['param_groups'][0]['lr'], loss.item())
            pbar.set_description(s)
        print("train accuracy", (train_correct / (len(train_dataset))))
        print("ce loss", loss_ce.item())
        # print("kd loss", kd_loss.item())
        scheduler.step()
        model.eval()
        correct = 0
        test_pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        print(('\n' + '%15s' * 3) % ('Epoch', 'Epochs', 'test_accuracy'))
        with torch.no_grad():
            total_test_loss = 0
            for i, (test_iamges, test_labels) in test_pbar:
                test_iamges = test_iamges.cuda()
                test_labels = test_labels.cuda()
                test_output = model(test_iamges).cuda()
                _, predicted = torch.max(test_output.data, 1)
                correct += (predicted == test_labels).sum().item()
                m = ('%15s' * 2 + '%15.4g' * 1) % (epoch, args.epochs, correct / len(test_dataset))
                test_pbar.set_description(m)
        test_acc = correct / len(test_dataset)
        print("test accuracy", test_acc)

        is_best = (test_acc >= best_acc)
        best_acc = max(test_acc, best_acc)
        print('Current best accuracy (top-1 accuracy):', best_acc)

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.job_dir)

        logger.info("=>Best accuracy {:.3f}".format(best_acc))  #


if __name__ == '__main__':
  main()
