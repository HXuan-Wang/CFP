from models.OctConv import *
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class Conv_Pool_BN(nn.Module):
    def __init__(self, in_high_channels,in_low_channels,out_high_channels,out_low_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_Pool_BN, self).__init__()
        self.conv = OctaveConv(in_high_channels,in_low_channels,out_high_channels,out_low_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(out_high_channels)
        self.bn_l = None if alpha_out == 0 else norm_layer(out_low_channels)

        self.pool_h = None if alpha_out == 1 else nn.MaxPool2d(kernel_size=2, stride=2,padding=0)
        self.pool_l = None if alpha_out == 0 else nn.MaxPool2d(kernel_size=2, stride=2,padding=0)

        self.act_h = nn.ReLU(inplace=True)
        self.act_l = nn.ReLU(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)

        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None

        x_h = self.act_h(x_h)
        x_l = self.act_l(x_l) if x_l is not None else None

        x_h = self.pool_h(x_h)
        x_l = self.pool_l(x_l) if x_l is not None else None
        return x_h, x_l

def adapt_channel(sparsity,num_blocks,num_out_channels):
    stage_oup_cprate = []
    for i in range(num_blocks):
        stage_oup_cprate +=[sparsity[i]]
    sparsity_channels=[]
    for i in range(len(num_out_channels)):
        sparsity_channels += [int((1-stage_oup_cprate[i])*num_out_channels[i])]
    return sparsity_channels

class DDDN(nn.Module):
    def __init__(self,num_classes=2, alpha=0.125,sparsity = None,num_block =
    [32, 32, 48, 64, 96, 96, 128, 128, 128], is_feature=False):
        super(DDDN, self).__init__()

        if (sparsity == None):
            self.sparsity = [0.] * 100
        else:
            self.sparsity = sparsity
        self.output_channels = num_block
        sparsity_channels = adapt_channel(self.sparsity, len(self.output_channels), self.output_channels)
        print(sparsity_channels)

        self.is_feature = is_feature

        self.conv1 = nn.Sequential(OrderedDict(
            [('conv',nn.Conv2d(3, sparsity_channels[0], 3, 1, 1, bias=False)),
             ('bn', nn.BatchNorm2d(sparsity_channels[0])),
             ('relu', nn.ReLU())
             ]
        ))

        self.conv2 = nn.Sequential(OrderedDict(
            [('conv',  nn.Conv2d(sparsity_channels[0], sparsity_channels[1], 3, 1, 1, bias=False)),
             ('bn', nn.BatchNorm2d(sparsity_channels[1])),
             ('relu', nn.ReLU()),
             ('maxpool',nn.MaxPool2d(kernel_size=2, stride=2))
             ]
        ))
        #in_high_channels,in_low_channels,out_high_channels,c
        in_low_channels = int(sparsity_channels[1] * 0)
        in_high_channels = sparsity_channels[1]-in_low_channels

        out_low_channels = int(sparsity_channels[2] * alpha)
        out_high_channels = sparsity_channels[2] - out_low_channels

        self.conv3 =(Conv_BN_ACT(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1,
                                              groups=1, bias=False, alpha_in=0, alpha_out=alpha))
        in_low_channels = int(sparsity_channels[2] * alpha)
        in_high_channels = sparsity_channels[2] - in_low_channels

        out_low_channels = int(sparsity_channels[3] * alpha)
        out_high_channels = sparsity_channels[3] - out_low_channels
        self.conv4 =(Conv_Pool_BN(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, \
                         alpha_in=alpha, alpha_out=alpha))
        in_low_channels = int(sparsity_channels[3] * alpha)
        in_high_channels = sparsity_channels[3] - in_low_channels

        out_low_channels = int(sparsity_channels[4] * alpha)
        out_high_channels = sparsity_channels[4] - out_low_channels
        self.conv5 = (Conv_BN_ACT(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, \
                        alpha_in=alpha, alpha_out=alpha))
        in_low_channels = int(sparsity_channels[4] * alpha)
        in_high_channels = sparsity_channels[4] - in_low_channels

        out_low_channels = int(sparsity_channels[5] * alpha)
        out_high_channels = sparsity_channels[5] - out_low_channels
        self.conv6 = (Conv_Pool_BN(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, \
                         alpha_in=alpha, alpha_out=alpha))
        in_low_channels = int(sparsity_channels[5] * alpha)
        in_high_channels = sparsity_channels[5] - in_low_channels

        out_low_channels = int(sparsity_channels[6] * alpha)
        out_high_channels = sparsity_channels[6] - out_low_channels
        self.conv7 = (Conv_BN_ACT(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, \
                        alpha_in=alpha, alpha_out=alpha))
        in_low_channels = int(sparsity_channels[6] * alpha)
        in_high_channels = sparsity_channels[6] - in_low_channels

        out_low_channels = int(sparsity_channels[7] * alpha)
        out_high_channels = sparsity_channels[7] - out_low_channels
        self.conv8 = (Conv_BN_ACT(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, \
                    alpha_in=alpha, alpha_out=alpha))
        in_low_channels = int(sparsity_channels[7] * alpha)
        in_high_channels = sparsity_channels[7] - in_low_channels

        out_low_channels = int(sparsity_channels[8] * 0)
        out_high_channels = sparsity_channels[8] - out_low_channels
        self.conv9 = (Conv_BN_ACT(in_high_channels,in_low_channels, out_high_channels,out_low_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False, \
                    alpha_in=alpha, alpha_out=0))

        self.avg =nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(sparsity_channels[8], 64)


        self.relu = nn.ReLU()

        self.output = nn.Linear(64, num_classes)


    def forward(self, x):
        feature=[]
        x = self.conv1(x)
        # feature.append(x)
        x = self.conv2(x)
        feature.append(x)
        x_h, x_l = self.conv3(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv4(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv5(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv6(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv7(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv8(x)
        feature.append(x_h)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv9(x)
        feature.append(x_h)
        # x = tuple([x_h, x_l])

        # x_h, x_l = self.features(x)
        x = self.avg(x_h)
        x = self.relu(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        #x = F.dropout(x, p=0.4, training=self.training)
        x = self.relu(x)

        x = self.output(x)
        if self.is_feature==True:
            return feature,x

       # x = F.dropout(x, p=0.4, training=self.training)

        return x
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


class DDDN_SC(nn.Module):
    def __init__(self,num_classes=2,alpha=0.125, sparsity_channels = None, is_feature=False):
        super(DDDN_SC, self).__init__()

        if (sparsity_channels == None):
            sparsity_channels = [32, 32, 6, 42, 8, 56, 8, 56, 12, 84, 12, 84, 12, 84, 12, 84, 16, 112, 16, 112, 16, 112, 16, 112, 128, 128]

        print(sparsity_channels)

        self.is_feature = is_feature

        self.conv1 = nn.Sequential(OrderedDict(
            [('conv', nn.Conv2d(3, sparsity_channels[0], 3, 1, 1, bias=False)),
             ('bn', nn.BatchNorm2d(sparsity_channels[0])),
             ('relu', nn.ReLU())
             ]
        ))

        self.conv2 = nn.Sequential(OrderedDict(
            [('conv', nn.Conv2d(sparsity_channels[0], sparsity_channels[1], 3, 1, 1, bias=False)),
             ('bn', nn.BatchNorm2d(sparsity_channels[1])),
             ('relu', nn.ReLU()),
             ('maxpool', nn.MaxPool2d(kernel_size=2, stride=2))
             ]
        ))

        # in_high_channels,in_low_channels,out_high_channels,c
        in_low_channels = sparsity_channels[1]
        in_high_channels = sparsity_channels[1]

        out_low_channels  = sparsity_channels[2]
        out_high_channels = sparsity_channels[3]

        self.conv3 = (
            Conv_BN_ACT(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                        padding=1,
                        groups=1, bias=False, alpha_in=0, alpha_out=alpha))
        in_low_channels =  sparsity_channels[2]
        in_high_channels = sparsity_channels[3]

        out_low_channels = sparsity_channels[4]
        out_high_channels = sparsity_channels[5]
        self.conv4 = (
            Conv_Pool_BN(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                         padding=1, groups=1, bias=False, \
                         alpha_in=alpha, alpha_out=alpha))
        in_low_channels = sparsity_channels[6]
        in_high_channels = sparsity_channels[7]

        out_low_channels = sparsity_channels[8]
        out_high_channels = sparsity_channels[9]
        self.conv5 = (
            Conv_BN_ACT(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                        padding=1, groups=1, bias=False, \
                        alpha_in=alpha, alpha_out=alpha))
        in_low_channels = sparsity_channels[10]
        in_high_channels = sparsity_channels[11]

        out_low_channels = sparsity_channels[12]
        out_high_channels =sparsity_channels[13]
        self.conv6 = (
            Conv_Pool_BN(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                         padding=1, groups=1, bias=False, \
                         alpha_in=alpha, alpha_out=alpha))
        in_low_channels = sparsity_channels[14]
        in_high_channels = sparsity_channels[15]

        out_low_channels =sparsity_channels[16]
        out_high_channels = sparsity_channels[17]
        self.conv7 = (
            Conv_BN_ACT(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                        padding=1, groups=1, bias=False, \
                        alpha_in=alpha, alpha_out=alpha))
        in_low_channels = sparsity_channels[18]
        in_high_channels = sparsity_channels[19]

        out_low_channels = sparsity_channels[20]
        out_high_channels = sparsity_channels[21]
        self.conv8 = (
            Conv_BN_ACT(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                        padding=1, groups=1, bias=False, \
                        alpha_in=alpha, alpha_out=alpha))
        in_low_channels = sparsity_channels[22]
        in_high_channels = sparsity_channels[23]

        out_low_channels = sparsity_channels[24]
        out_high_channels = sparsity_channels[25]
        self.conv9 = (
            Conv_BN_ACT(in_high_channels, in_low_channels, out_high_channels, out_low_channels, kernel_size=3, stride=1,
                        padding=1, groups=1, bias=False, \
                        alpha_in=alpha, alpha_out=0))

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(sparsity_channels[25], 64)

        self.relu = nn.ReLU()

        self.output = nn.Linear(64, num_classes)


    def forward(self, x):
        feature=[]
        x = self.conv1(x)
        # feature.append(x)
        x = self.conv2(x)
        feature.append(x)
        x_h, x_l = self.conv3(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv4(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv5(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv6(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv7(x)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv8(x)
        feature.append(x_h)
        x = tuple([x_h, x_l])
        x_h, x_l = self.conv9(x)
        feature.append(x_h)
        # x = tuple([x_h, x_l])

        # x_h, x_l = self.features(x)
        x = self.avg(x_h)
        x = self.relu(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        #x = F.dropout(x, p=0.4, training=self.training)
        x = self.relu(x)

        x = self.output(x)
        if self.is_feature==True:
            return feature,x

       # x = F.dropout(x, p=0.4, training=self.training)

        return x
    def initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()



def DDDN(sparsity_channels=None,sparsity=None,num_classes=10, original=False):
    if original:
        return DDDN(num_classes=num_classes, sparsity=sparsity)
    else:
        return DDDN_SC(sparsity_channels=sparsity_channels,num_classes=num_classes)




