import numpy as np
import torch
import os
import argparse
import scipy
import scipy.cluster.hierarchy as sch
import skimage.measure
parser = argparse.ArgumentParser(description='Calculate cluster')

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn','mobilenet_v2','DDDN','resnet_50','RepVGG-A0'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--num_layers',
    type=int,
    default=21,
    help='conv layers in the model')

parser.add_argument(
    '--scale',
    type=float,
    default=2,)

parser.add_argument(
    '--distance_dir',
    type=str,
    default='./Distance_RepVGG-A0',
    help='feature maps dir')

parser.add_argument(
    '--gpu',
    type=str,
    default='1',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# criterionKD = Entropy(2).cuda()

def mean_repeat_ci(repeat, num_layers):
    layer_ci_mean_total = []
    for j in range(num_layers):
        path_conv = "{0}/dis_conv{1}.npy".format(str(args.distance_dir), str(j+1))
        conv_res = np.array(np.load(path_conv))

        print(conv_res.shape)

        disMat = sch.distance.squareform(conv_res)
        Z = sch.linkage(disMat, method='average')
        print("max",max(disMat))
        print("min", min(disMat))
        # thresold = (max(disMat)-min(disMat))/args.scale #-min(disMat)
        # f = sch.fcluster(Z, t=thresold, criterion='distance')  # t=thresold, criterion='distance'
        # if (max(f)>4):
        #     f = sch.fcluster(Z, t=4, criterion='maxclust')
        # print(max(f))
        f = sch.fcluster(Z, t=2, criterion='maxclust')
        print(np.bincount(f))
        layer_ci_mean_total.append(f)

    return np.array(layer_ci_mean_total)


def main():

    num_layers = args.num_layers
    save_path = 'HC_' + args.arch
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ci = mean_repeat_ci(0, num_layers)
    for i in range(num_layers):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/ci_conv{0}.npy".format(str(i + 1)), ci[i])



if __name__ == '__main__':
    main()



