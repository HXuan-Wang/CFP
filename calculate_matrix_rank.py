import numpy as np
import torch
import os
import argparse
import skimage.measure
parser = argparse.ArgumentParser(description='Calculate Rank')

parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('vgg_16_bn','mobilenet_v2','DDDN','resnet_50','RepVGG-A0'),
    help='architecture to calculate feature maps')

parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='repeat times')

parser.add_argument(
    '--num_layers',
    type=int,
    default=12,
    help='conv layers in the model')

parser.add_argument(
    '--feature_map_dir',
    type=str,
    default='./feature_maps',
    help='feature maps dir')
parser.add_argument(
    '--gpu',
    type=str,
    default='5',
    help='Select gpu to use')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



def ci_score(path_conv):
    conv_output = torch.tensor(np.round(np.load(path_conv), 4)).cuda()
    ci = torch.zeros(conv_output.shape[0],conv_output.shape[1]).cuda()
    a = conv_output.shape[0]
    b = conv_output.shape[1]
    for i in range(a):
        for j in range(b):
            features  = torch.tensor(conv_output[i, j, :, :]).cuda()
            ci[i][j] = torch.matrix_rank(features)

    return ci

def mean_repeat_ci(repeat, num_layers):
    layer_ci_mean_total = []
    for j in range(num_layers):

        repeat_ci_mean = []
        for i in range(repeat):
            index = j * repeat + i + 1
            print("index", index, "start")
            # add
            path_conv = "./{0}/{1}_repeat5/conv_feature_map_tensor({2}).npy".format(str(args.feature_map_dir),str(args.arch), str(index))

            batch_ci = ci_score(path_conv).cpu().numpy()
            single_repeat_ci_mean = np.mean(batch_ci, axis=0)
            print(single_repeat_ci_mean.shape)
            repeat_ci_mean.append(single_repeat_ci_mean)
            print("index", index, "finished")

        layer_ci_mean = np.mean(repeat_ci_mean, axis=0)
        layer_ci_mean_total.append(layer_ci_mean)
    return np.array(layer_ci_mean_total)



def main():
    repeat = args.repeat

    num_layers = args.num_layers
    save_path = 'Rank_' + args.arch
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    ci = mean_repeat_ci(repeat, num_layers)

    for i in range(num_layers):
        print(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/rank_conv{0}.npy".format(str(i + 1)), ci[i])


if __name__ == '__main__':
    main()



