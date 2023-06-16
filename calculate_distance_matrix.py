import numpy as np
import torch
import os
import argparse
import skimage.measure
parser = argparse.ArgumentParser(description='Calculate distane matrix')

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


def distance_matrix(path_conv):

    conv_output = torch.tensor(np.load(path_conv)).cuda()
    conv_output = conv_output.permute(1, 0, 2, 3).cuda()
    distance = torch.zeros([conv_output.shape[0], conv_output.shape[0]]).cuda()
    for i in range(conv_output.shape[0]):
        for j in range(i + 1, conv_output.shape[0]):
            total_norm = np.zeros(conv_output.shape[1])
            for index in range(conv_output.shape[1]):
                len = conv_output.shape[2]
                ori_data = (conv_output[i, index, :, :].clone()).cuda()
                ori_norm = torch.norm(ori_data, p=2).item()
                reduced_data = (conv_output[j, index, :, :].clone()).cuda()
                reduced_norm = torch.norm(reduced_data, p=2).item()
                res = np.absolute(ori_norm - reduced_norm)
                res /= len
                total_norm[index] = res
            norms = np.mean(total_norm, axis=0)
            distance[i, j] = norms
            distance[j, i] = distance[i, j]

    return distance

def mean_repeat_distance(repeat, num_layers):
    layer_distance_mean_total = []
    for j in range(num_layers):

        repeat_distance_mean = []
        for i in range(repeat):
            index = j * repeat + i + 1
            print("index", index, "start")
            # add
            path_conv = "./{0}/{1}_repeat5/conv_feature_map_tensor({2}).npy".format(str(args.feature_map_dir),str(args.arch), str(index))

            batch_distance = distance_matrix(path_conv).cpu().numpy()
            single_repeat_distance_mean = batch_distance
            repeat_distance_mean.append(single_repeat_distance_mean)
            print("index", index, "finished")

        layer_distance_mean = np.mean(repeat_distance_mean, axis=0)
        layer_distance_mean_total.append(layer_distance_mean)
    return np.array(layer_distance_mean_total)



def main():
    repeat = args.repeat

    num_layers = args.num_layers
    save_path = 'Distance_' + args.arch
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dis = mean_repeat_distance(repeat, num_layers)


    for i in range(num_layers):
        print(i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + "/dis_conv{0}.npy".format(str(i + 1)), dis[i])


if __name__ == '__main__':
    main()


