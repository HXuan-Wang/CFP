import os
import argparse
import numpy as np
import torch
from thop import profile
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from models.resnet_imagenet import resnet_50
from models.VGG16 import vgg_16_bn
from models.repvgg import RepVGGA0
from models.DDDN import *
from dataset import *
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(description='Calculate Feature Maps')

parser.add_argument(
    '--arch',
    type=str,
    default='resnet_50',
    choices=('vgg_16_bn','mobilenet_v2','DDDN','resnet_50','RepVGGA0'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--dataset',
    type=str,
    default='GC10-DET',
    choices=('GC10-DET','NEU-CLS','ELPV','X-SDD'),
    help='cifar10 or imagenet')

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='',
    help='dir for the pretriained model to calculate feature maps')

parser.add_argument(
    '--batch_size',
    type=int,
    default=128,
    help='batch size for one batch.')
parser.add_argument(
    '--input_size',
    type=int,
    default=224,
    help='size of input image')
parser.add_argument(
    '--save_dir',
    type=str,
    default='./feature_maps/',
    help='dataset path')
parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='the number of different batches for calculating feature maps.')
parser.add_argument(
    '--num_classes',
    type=int,
    default=10,
    help='size of input image')
parser.add_argument(
    '--gpu',
    type=str,
    default='7',
    help='gpu id')

args = parser.parse_args()

# gpu setting
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes=args.num_classes
save_path=args.save_dir
# prepare data

input_size = args.input_size
if args.dataset=='GC10-DET':
    test_transform = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ])
    train_data = MyDataset(fname=os.getcwd(), transform=test_transform, train=True)
elif args.dataset=='X-SDD':
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    train_data = MyDataset(fname=os.getcwd(), transform=test_transform, train=True)
elif args.dataset=='NEU-CLS':
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    train_data = MyDataset(fname=os.getcwd(), transform=test_transform, train=True)

elif args.dataset=='ELPV':
    test_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
    ])
    train_data = MyDataset(fname=os.getcwd() + "/elpv-dataset-master/utils", transform=test_transform, train=True)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)

# Model

model = eval(args.arch)(num_classes=num_classes,sparsity_channels=None,original=True).cuda()

if args.arch=="DDDN" and args.dataset=='GC10-DET':
    input_image = torch.randn(1, 3, input_size, input_size).cuda()
    flops, params = profile(model, inputs=(input_image,))


# Load pretrained model.
print('Loading Pretrained Model...')

checkpoint = torch.load(args.pretrain_dir)
if args.arch=='RepVGGA0':
    model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint['state_dict'])

print('Loading Pretrained Model finished...')

conv_index = torch.tensor(1)

def get_feature_hook(self, input, output):
    global conv_index

    if not os.path.isdir(save_path + args.arch + '_repeat%d' % (args.repeat)):
        os.makedirs(save_path + args.arch + '_repeat%d' % (args.repeat))
    np.save(save_path + args.arch + '_repeat%d' % (args.repeat) + '/conv_feature_map_'+ str(conv_index) + '.npy',
            output.cpu().numpy())

    conv_index += 1

def inference():
    model.eval()
    repeat = args.repeat
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #use 5 batches to get feature maps.
            if batch_idx >= repeat:
               break
            inputs, targets = inputs.to(device), targets.to(device)

            model(inputs)

if args.arch=='vgg_16_bn':
    relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]

    start = time.time()
    for i, cov_id in enumerate(relucfg):
        cov_layer = model.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

elif args.arch=='resnet_50':
    cov_layer = eval('model.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    # ResNet50 per bottleneck
    for i in range(4):
        block = eval('model.layer%d' % (i + 1))
        for j in range(model.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

            if j==0:
                cov_layer = block[j].relu3
                handler = cov_layer.register_forward_hook(get_feature_hook)
                inference()
                handler.remove()

elif args.arch=='RepVGGA0':
    num_blocks = [2, 4, 14, 1]
    cov_layer = eval('model.stage0.nonlinearity')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()
    for i in range(4):
        block = eval('model.stage%d' % (i + 1))
        for j in range(num_blocks[i]):
            cov_layer = block[j].nonlinearity
            print(cov_layer)
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()

elif args.arch=='DDDN':
    conv_layer = eval('model.conv1.relu')
    print(conv_layer)
    handler = conv_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()
    print("finished")

    conv_layer = eval('model.conv2.relu')
    print(conv_layer)
    handler = conv_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()
    print("finished")

    block = eval('model.conv3')
    conv_layer = block.conv.convh_l
    print(conv_layer)
    handler = conv_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    conv_layer = block.conv.convh_h
    print(conv_layer)
    handler = conv_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    for i in range(4, 9):
        block = eval('model.conv%d' % (i))
        conv_layer = block.conv.convl_l
        print(conv_layer)
        handler = conv_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        conv_layer = block.conv.convl_h
        print(conv_layer)
        handler = conv_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        conv_layer = block.conv.convh_l
        print(conv_layer)
        handler = conv_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

        conv_layer = block.conv.convh_h
        print(conv_layer)
        handler = conv_layer.register_forward_hook(get_feature_hook)
        inference()
        handler.remove()

    block = eval('model.conv9')
    conv_layer = block.conv.convl_h
    print(conv_layer)
    handler = conv_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    conv_layer = block.conv.convh_h
    print(conv_layer)
    handler = conv_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

elif args.arch=='mobilenet_v2':
    cov_layer = eval('model.features[0]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    inference()
    handler.remove()

    cnt=1
    for i in range(1,19):
        if i==1:
            block = eval('model.features[%d].conv' % (i))
            relu_list=[2,4]
        elif i==18:
            block = eval('model.features[%d]' % (i))
            relu_list=[2]
        else:
            block = eval('model.features[%d].conv' % (i))
            relu_list = [2,5,7]

        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            inference()
            handler.remove()