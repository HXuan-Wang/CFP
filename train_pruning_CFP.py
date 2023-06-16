
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

from dataset import *
import utils.common as utils
from models.VGG16 import vgg_16_bn
from models.DDDN import DDDN
from models.repvgg import RepVGGA0
from torch.utils.data.sampler import WeightedRandomSampler
from st import *


parser = argparse.ArgumentParser("ELPV training")

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn','mobilenet_v2','DDDN','resnet_50','RepVGGA0'))

parser.add_argument(
    '--dataset',
    type=str,
    default='GC10-DET',
    choices=('GC10-DET','NEU-CLS','ELPV','X-SDD'))

parser.add_argument(
    '--job_dir',
    type=str,
    default='./pruned_models')
parser.add_argument(
    '--batch_size',
    type=int,
    default=128)
parser.add_argument(
    '--epochs',
    type=int,
    default=300)
parser.add_argument('-lr', type=float, default=0.00005, help=' learning rate')  # 0.001
parser.add_argument(
    '--resume',
    default=False)

parser.add_argument(
    '--use_pretrain',
    default=True)

parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='./original_models/VGG16_bn.pth.tar')

parser.add_argument(
    '--teacher_dir',
    type=str,
    default='./original_models/VGG16_Balance.pth.tar')

parser.add_argument(
    '--rank_conv_prefix',
    type=str,
    default='')
parser.add_argument(
    '--hierarchical_conv_prefix',
    type=str,
    default='')

parser.add_argument(
    '--compress_rate',
    type=str,
    default='[0.]+[0.4]*12+[0.5]*12')
parser.add_argument(
    '--input_size',
    type=int,
    default=224)
parser.add_argument(
    '--num_classes',
    type=int,
    default=10)
parser.add_argument(
    '--test_only',
    default=False
   )

parser.add_argument(
    '--test_model_dir',
    type=str,
    default='./models2/model_best.pth.tar')
parser.add_argument(
    '--gpu',
    type=str,
    default='2')
parser.add_argument('--T', type=float, default=2)


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

def load_sparse_channels(index = 0):
    path_conv = "{0}/ci_conv{1}.npy".format(str(args.hierarchical_conv_prefix), str(index))
    hierarchical = np.load(path_conv)
    num = max(hierarchical)
    total_channels = []
    for j in range(1, num + 1):
        channels = []
        for index, label in enumerate(hierarchical):
            if label == j:
                channels.append(index)
        total_channels.append(channels)
    return total_channels,num

def load_DDDN_model(model,oristate_dict,stage_oup_cprate,output_channnels):
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer
    last_h_h_select_index = None  # h_h Conv index selected in the previous layer
    last_h_l_select_index = None  # h_l Conv index selected in the previous layer
    last_l_h_select_index = None  # l_h Conv index selected in the previous layer
    last_l_l_select_index = None  # l_l Conv index selected in the previous layer

    cnt = 0
    prefix = args.rank_conv_prefix + '/rank_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            if "conv1" in name:
                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name_base + name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                if orifilter_num != currentfilter_num:
                    cov_id = cnt
                    logger.info(name+' original conv1 loading rank from: ' + prefix + str(cov_id) + subfix)
                    rank = np.load(prefix + str(cov_id) + subfix)
                    total_channels, num = load_sparse_channels(cov_id)
                    select_index = []
                    for i in range(num):
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                    select_index = np.array(select_index)
                    select_index.sort()


                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base + name + '.weight'][index_i] = \
                                oristate_dict[name + '.weight'][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + name + '.weight'][i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    state_dict[name_base + name + '.weight'] = oriweight
                    last_select_index = None
            if "conv2" in name:
                cnt += 1
                oriweight = oristate_dict[name + '.weight']
                curweight = state_dict[name_base + name + '.weight']
                orifilter_num = oriweight.size(0)
                currentfilter_num = curweight.size(0)
                if orifilter_num != currentfilter_num:
                    cov_id = cnt
                    logger.info(name+'original conv2 loading rank from: ' + prefix + str(cov_id) + subfix)
                    rank = np.load(prefix + str(cov_id) + subfix)
                    total_channels, num = load_sparse_channels(cov_id)
                    select_index = []
                    for i in range(num):
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                    select_index = np.array(select_index)
                    select_index.sort()


                    if last_select_index is not None:
                        for index_i, i in enumerate(select_index):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        for index_i, i in enumerate(select_index):
                            state_dict[name_base + name + '.weight'][index_i] = \
                                oristate_dict[name + '.weight'][i]

                    last_select_index = select_index

                elif last_select_index is not None:
                    for i in range(orifilter_num):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + name + '.weight'][i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    state_dict[name_base + name + '.weight'] = oriweight
                    last_select_index = None

            if "conv3" in name:
                if 'h_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_select_index = select_index

                        last_l_l_select_index = select_index
                        last_l_h_select_index = select_index

                    elif last_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]
                        last_h_h_select_index = select_index
                        last_h_l_select_index = select_index
                        last_select_index = select_index

                        # last_l_h_select_index = select_index

                    elif last_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_select_index = None

            if "conv4" in name:
                if 'l_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        cur_channl_sum=0
                        for i in range(num-1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum+=cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i=num-1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)

                        cur = output_channnels[cov_id-1]-cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])

                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_l_select_index = select_index

                    elif last_l_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_l_select_index = None
                if 'l_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_h_select_index = select_index

                    elif last_l_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_h_select_index = None
                if 'h_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_l_select_index = select_index

                    elif last_h_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_l_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_h_select_index = select_index

                    elif last_h_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_h_select_index = None
            if "conv5" in name:
                if 'l_l' in name:
                    cnt += 1

                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_l_select_index = select_index

                    elif last_l_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_l_select_index = None
                if 'l_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info(name+' loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_l_select_index = select_index

                    elif last_h_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_l_select_index = None
                if 'h_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_h_select_index = select_index

                    elif last_l_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_h_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_h_select_index = select_index

                    elif last_h_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_h_select_index = None

            if "conv6" in name:
                if 'l_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_l_select_index = select_index

                    elif last_l_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_l_select_index = None
                if 'l_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_h_select_index = select_index

                    elif last_l_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_h_select_index = None
                if 'h_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_l_select_index = select_index

                    elif last_h_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_l_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_h_select_index = select_index

                    elif last_h_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_h_select_index = None

            if "conv7" in name:
                if 'l_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_l_select_index = select_index

                    elif last_l_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_l_select_index = None
                if 'l_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_l_select_index = select_index

                    elif last_h_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_l_select_index = None
                if 'h_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_h_select_index = select_index

                    elif last_l_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_h_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_h_select_index = select_index

                    elif last_h_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_h_select_index = None

            if "conv8" in name:
                if 'l_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_l_select_index = select_index

                    elif last_l_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_l_select_index = None
                if 'l_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_l_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_l_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_l_h_select_index = select_index

                    elif last_l_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_l_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_l_h_select_index = None
                if 'h_l' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_l loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_l_select_index = select_index

                    elif last_h_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_l_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_h_select_index = select_index

                    elif last_h_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_h_select_index = None

            if "conv9" in name:

                if 'l_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('l_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_l_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_l_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_l_select_index = select_index

                    elif last_h_l_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_l_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_l_select_index = None
                if 'h_h' in name:
                    cnt += 1
                    oriweight = oristate_dict[name + '.weight']
                    curweight = state_dict[name_base + name + '.weight']
                    orifilter_num = oriweight.size(0)
                    currentfilter_num = curweight.size(0)
                    if orifilter_num != currentfilter_num:
                        cov_id = cnt
                        logger.info('h_h loading rank from: ' + prefix + str(cov_id) + subfix)
                        rank = np.load(prefix + str(cov_id) + subfix)
                        total_channels, num = load_sparse_channels(cov_id)
                        select_index = []
                        for i in range(num - 1):
                            data_rank = []
                            for number in (total_channels[i]):
                                data_rank.append(rank[number])
                            ori = len(data_rank)
                            cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)
                            cur_channl_sum += cur

                            sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                            for m in sparsity_select_index:
                                select_index.append(total_channels[i][m])
                        i = num - 1
                        data_rank = []
                        for number in (total_channels[i]):
                            data_rank.append(rank[number])
                        ori = len(data_rank)
                        cur = output_channnels[cov_id-1] - cur_channl_sum

                        sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                        for m in sparsity_select_index:
                            select_index.append(total_channels[i][m])
                        select_index = np.array(select_index)
                        select_index.sort()


                        if last_h_h_select_index is not None:
                            for index_i, i in enumerate(select_index):
                                for index_j, j in enumerate(last_h_h_select_index):
                                    state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                        oristate_dict[name + '.weight'][i][j]
                        else:
                            for index_i, i in enumerate(select_index):
                                state_dict[name_base + name + '.weight'][index_i] = \
                                    oristate_dict[name + '.weight'][i]

                        last_h_h_select_index = select_index

                    elif last_h_h_select_index is not None:
                        for i in range(orifilter_num):
                            for index_j, j in enumerate(last_h_h_select_index):
                                state_dict[name_base + name + '.weight'][i][index_j] = \
                                    oristate_dict[name + '.weight'][i][j]
                    else:
                        state_dict[name_base + name + '.weight'] = oriweight
                        last_h_h_select_index = None

    model.load_state_dict(state_dict)


def load_repvgg_model(model,oristate_dict,stage_oup_cprate):
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer
    print("stage_oup_cprate",stage_oup_cprate)
    cnt = 0
    prefix = args.rank_conv_prefix + '/rank_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            cnt += 1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base + name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)
            if orifilter_num != currentfilter_num:
                cov_id = cnt
                logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                rank = np.load(prefix + str(cov_id) + subfix)
                total_channels, num = load_sparse_channels(cov_id)
                select_index = []
                for i in range(num):
                    data_rank = []
                    for number in (total_channels[i]):
                        data_rank.append(rank[number])
                    ori = len(data_rank)
                    cur = int((1 - stage_oup_cprate[cov_id-1]) * ori)

                    sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                    for m in sparsity_select_index:
                       select_index.append(total_channels[i][m])
                select_index = np.array(select_index)
                select_index.sort()


                if last_select_index is not None:

                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:

                    for index_i, i in enumerate(select_index):
                        state_dict[name_base + name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index


            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base + name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base + name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def load_vgg_model(model, oristate_dict,stage_oup_cprate):
    state_dict = model.state_dict()
    last_select_index = None  # Conv index selected in the previous layer
    print("stage_oup_cprate", stage_oup_cprate)

    cnt = 0
    prefix = args.rank_conv_prefix + '/rank_conv'
    subfix = ".npy"
    for name, module in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, nn.Conv2d):
            cnt += 1
            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name_base + name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)
            if orifilter_num != currentfilter_num:
                cov_id = cnt
                logger.info('loading ci from: ' + prefix + str(cov_id) + subfix)
                rank = np.load(prefix + str(cov_id) + subfix)
                total_channels, num = load_sparse_channels(cov_id)
                select_index = []
                for i in range(num):
                    data_rank = []
                    for number in (total_channels[i]):
                        data_rank.append(rank[number])
                    ori = len(data_rank)
                    cur = int((1 - stage_oup_cprate[cov_id - 1]) * ori)

                    sparsity_select_index = np.argsort(data_rank)[ori - cur:]
                    for m in sparsity_select_index:
                        select_index.append(total_channels[i][m])
                select_index = np.array(select_index)
                select_index.sort()


                if last_select_index is not None:

                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name_base + name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:

                    for index_i, i in enumerate(select_index):
                        state_dict[name_base + name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index


            elif last_select_index is not None:
                for i in range(orifilter_num):
                    for index_j, j in enumerate(last_select_index):
                        state_dict[name_base + name + '.weight'][i][index_j] = \
                            oristate_dict[name + '.weight'][i][j]
            else:
                state_dict[name_base + name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

def get_channels(ori_channels=None,sparsity=None):

    stage_oup_cprate = []
    stage_oup_cprate += [sparsity[0]]
    for i in range(len(ori_channels)-2):
        stage_oup_cprate += [sparsity[i+1]]
    stage_oup_cprate += [0.]
    sparsity_channels = []
    indexs = 0
    sparsity_single_channels = []
    all_single_channels = []
    for i in range(len(ori_channels)-1):
        path_conv = "{0}/ci_conv{1}.npy".format(str(args.hierarchical_conv_prefix), str(indexs + 1))
        hierarchical = np.load(path_conv)
        num = max(hierarchical)

        channels = []
        for j in range(1, num + 1):
            channels_index = 0
            for index, label in enumerate(hierarchical):
                if label == j:
                    channels_index += 1
            channels.append(channels_index)
        print("hierarchical channel number",channels)

        target_channels=0
        for x in channels:
            all_single_channels.append(x)
            single_channels = int((1 - stage_oup_cprate[i]) * x)
            sparsity_single_channels.append(single_channels)
            target_channels += single_channels
        sparsity_channels += [target_channels]
        indexs+=1
    sparsity_channels += [ori_channels[-1]]

    print("get_channels sparsity_channels",sparsity_channels)

    print("get_channels sparsity_single_channels", sparsity_single_channels)

    print("get_channels stage_oup_cprate", stage_oup_cprate)

    return sparsity_channels,sparsity_single_channels,stage_oup_cprate


def get_target_channels(args,compress_rate):
    if args.arch == 'RepVGGA0':
        num_blocks = [2, 4, 14, 1]
        width_multiplier = [0.75, 0.75, 0.75, 2.5]
        original_channels = [min(64, int(64 * width_multiplier[0]))] + num_blocks[0] * [
            (int(64 * width_multiplier[0]))] + \
                            num_blocks[1] * [
                                (int(128 * width_multiplier[1]))] \
                            + num_blocks[2] * [int(256 * width_multiplier[2])] + num_blocks[3] * [
                                int(512 * width_multiplier[3])]
        output_channnels, out_single_channels, stage_oup_cprate = get_channels(ori_channels=original_channels,
                                                                               sparsity=compress_rate)
        return output_channnels, out_single_channels, stage_oup_cprate
    elif args.arch == 'vgg_16_bn':
        original_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
        output_channnels, out_single_channels, stage_oup_cprate = get_channels(ori_channels=original_channels,
                                                                               sparsity=compress_rate)
        return output_channnels, out_single_channels, stage_oup_cprate
    elif args.arch == 'DDDN':
        original_channels = [32, 32, 6, 42, 8, 56, 8, 56, 12, 84, 12, 84, 12, 84, 12, 84, 16, 112, 16, 112, 16, 112, 16,
         112, 128, 128]

        output_channnels, out_single_channels, stage_oup_cprate = get_channels(ori_channels=original_channels,
                                                                               sparsity=compress_rate)
        for i in range(4,21,4):
            if(output_channnels[i]!=output_channnels[i+2]):
                output_channnels[i + 2]=output_channnels[i]
            if (output_channnels[i+1] != output_channnels[i + 3]):
                output_channnels[i + 3] = output_channnels[i+1]

        return output_channnels, out_single_channels, stage_oup_cprate
    return None,None,None

def main():

    cudnn.benchmark = True
    cudnn.enabled=True
    logger.info("args = %s", args)

    if args.compress_rate:
        import re
        cprate_str = args.compress_rate
        cprate_str_list = cprate_str.split('+')
        pat_cprate = re.compile(r'\d+\.\d*')
        pat_num = re.compile(r'\*\d+')
        cprate = []
        for x in cprate_str_list:
            num = 1
            find_num = re.findall(pat_num, x)
            if find_num:
                assert len(find_num) == 1
                num = int(find_num[0].replace('*', ''))
            find_cprate = re.findall(pat_cprate, x)
            assert len(find_cprate) == 1
            cprate += [float(find_cprate[0])] * num

        compress_rate = cprate

    num_classes = args.num_classes
    # load model
    logger.info('compress_rate:' + str(compress_rate))
    output_channnels, out_single_channels, stage_oup_cprate= get_target_channels(args,compress_rate)

    print("output_channnels",output_channnels)

    if args.arch=='vgg_16_bn':
        original_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        index = 0
        for i in range(len(original_cfg) - 1):
            if original_cfg[i] != 'M':
                original_cfg[i] = int(output_channnels[index])
                index += 1
        output_channnels=original_cfg

    model = eval(args.arch)(num_classes=num_classes,sparsity_channels=output_channnels)
    model = model.cuda()
    logger.info(model)

    input_size = args.input_size
    input_image_size=input_size

    if args.arch == "DDDN":
        input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
        flops, params = profile(model, inputs=(input_image,))
        logger.info('Params: %.2f M' % (params / 1e6))
        logger.info('Flops: %.2f G' % (flops / 1e9))

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
        train_data = MyDataset(fname=os.getcwd()+ "/elpv-dataset-master/utils", transform=train_transform, train=True)
        test_data = MyDataset(fname=os.getcwd()+ "/elpv-dataset-master/utils", transform=test_transform, train=False)

    # load training data
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=4)

    criterionKL = SoftTarget(args.T).cuda()


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
            print("accuracy, precision,recall,F1",test_acc.compute(),test_precision.compute(),test_recall.compute(),test_F1score.compute())

            input = torch.randn(64, 3, input_size, input_size, dtype=torch.float32)  # args.batch_size
            input = input.cuda()
            for i in range(50):
                input = input.cuda()
                output = model(input)
                torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            times = []
            with torch.no_grad():
                for i in range(100):
                    torch.cuda.synchronize()
                    start_evt.record()
                    test_output = model(input).cuda()
                    end_evt.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_evt.elapsed_time(end_evt)
                    times.append(elapsed_time)
                print("Infer time (ms/image)", np.mean(times) / 64)

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
                                 weight_decay=5e-4,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True, eta_min=1e-6)

    start_epoch = 0
    best_acc= 0

    # load the checkpoint if it exists

    checkpoint_dir = os.path.join(args.job_dir, 'model_best.pth.tar')


    origin_model = eval(args.arch)(num_classes=num_classes, sparsity_channels=None, original=True)

    origin_model = origin_model.cuda()

    # origin_model = origin_model.cuda()

    input_image_size = input_size
    if args.arch == "DDDN" or (args.arch == 'vgg_16_bn' and args.dataset=='X-SDD'):
        input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
        flops, params = profile(origin_model, inputs=(input_image,))
        logger.info('Params: %.2f M' % (params / 1e6))
        logger.info('Flops: %.2f G' % (flops / 1e9))

    model_t = eval(args.arch)(num_classes=num_classes, sparsity_channels=None, original=True)
    model_t = model_t.cuda()
    if args.arch == 'DDDN' :
        input_image_size = input_size
        input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
        flops, params = profile(model_t, inputs=(input_image,))
        logger.info('Params: %.2f M' % (params / 1e6))
        logger.info('Flops: %.2f G' % (flops / 1e9))
    ckpt_t = torch.load(args.teacher_dir)
    pt_t = ckpt_t if args.arch == 'RepVGGA0' else ckpt_t['state_dict']
    model_t.load_state_dict(pt_t)



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
    else:
        if args.use_pretrain:
            logger.info('resuming from pretrain model')

            ckpt = torch.load(args.pretrain_dir)
            m = ckpt if args.arch=='RepVGGA0' else ckpt['state_dict']

            origin_model.load_state_dict(m)
            origin_model.eval()
            test_pbar = tqdm(enumerate(val_loader), total=len(val_loader))
            correct = 0
            print(('\n' + '%15s' * 3) % ('Epoch', 'Epochs', 'test_accuracy'))
            with torch.no_grad():
                for i, (test_iamges, test_labels) in test_pbar:
                    test_iamges = test_iamges.cuda()
                    test_labels = test_labels.cuda()
                    test_output = origin_model(test_iamges).cuda()
                    _, predicted = torch.max(test_output.data, 1)
                    correct += (predicted == test_labels).sum().item()
                    m = ('%15s' * 2 + '%15.4g' * 1) % (0, args.epochs, correct / len(test_data))
                    test_pbar.set_description(m)
            test_acc = correct / len(test_data)
            print("original test accuracy", test_acc)

            oristate_dict = origin_model.state_dict()

            if args.arch == 'RepVGGA0':
                load_repvgg_model(model, oristate_dict, stage_oup_cprate)


            elif args.arch == 'vgg_16_bn':
                load_vgg_model(model, oristate_dict, stage_oup_cprate)

            elif args.arch == 'DDDN':
                print("stage_oup_cprate",stage_oup_cprate)
                load_DDDN_model(model, oristate_dict, stage_oup_cprate,output_channnels)

        else:
            logger('training from scratch')

    # adjust the learning rate according to the checkpoint


    for epoch in range(start_epoch,args.epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        model.train()
        print(('\n' + '%15s' * 4) % ('Epoch', 'Epochs', 'lr', 'train_loss'))
        train_correct = 0
        total_train_loss = 0
        for i, (images, labels) in pbar:
            images = images.cuda()
            labels = labels.long().cuda()

            with torch.no_grad():
                output_t = model_t(images)
                output_o = origin_model(images)

            output_s = model(images)

            kd_loss_t = criterionKL(output_s, output_t.detach()).cuda()
            kd_loss_o = criterionKL(output_s, output_o.detach()).cuda()
            l = np.cos((epoch / args.epochs) * (math.pi / 2))

            loss = l * kd_loss_o + (1 - l) * kd_loss_t

            total_train_loss += loss
            _, predicted = torch.max(output_s.data, 1)
            train_correct += (predicted == labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s = ('%15s' * 2 + '%15.4g' * 2) % (
                epoch, args.epochs, optimizer.state_dict()['param_groups'][0]['lr'], loss.item())
            pbar.set_description(s)
        print("train accuracy", (train_correct / (len(train_data))))
        print("loss", loss.item())
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
                m = ('%15s' * 2 + '%15.4g' * 1) % (epoch, args.epochs, correct / len(test_data))
                test_pbar.set_description(m)
        test_acc = correct / len(test_data)
        print("test accuracy", test_acc)

        is_best = test_acc >= best_acc
        best_acc = max(test_acc, best_acc)
        print('Current best accuracy (top-1 accuracy):', best_acc)

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
            }, is_best, args.job_dir)

        logger.info("=>Best accuracy {:.3f}".format(best_acc))#

if __name__ == '__main__':
  main()
