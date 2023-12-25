# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :MobileNet
# @File     :OctConv
# @Date     :2020/10/30 20:55
# @Author   :Wanghaoxuan
# @Software :PyCharm
-------------------------------------------------
"""

import torch
import torch.nn as nn
import math
class OctaveConv(nn.Module):
    def __init__(self,in_high_channels,in_low_channels,out_high_channels,out_low_channels,kernel_size,alpha_in=0.5,alpha_out=0.5,
                 stride = 1,padding=1,dilation=1,groups=1,bias=False):
        super(OctaveConv,self).__init__()

        self.downsample = nn.AvgPool2d(kernel_size=(2,2),stride=2,padding=0)
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        self.is_dw = groups == in_high_channels+in_low_channels # 如果groups=in_channels 即使用可分离卷积，则没有高低频的交换部分

        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out

        self.convl_l = None if alpha_in == 0 or alpha_out == 0 else \
            nn.Conv2d(in_low_channels,out_low_channels,
                # int(alpha_in * in_channels), int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, math.ceil(alpha_in * groups), bias)
        self.convl_h = None if alpha_in == 0 or alpha_out == 1 or self.is_dw else \
            nn.Conv2d(
                in_low_channels,out_high_channels,
                # int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, groups, bias)
        self.convh_l = None if alpha_in == 1 or alpha_out == 0 or self.is_dw else \
            nn.Conv2d(in_high_channels,out_low_channels,
                # in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, groups, bias)
        self.convh_h = None if alpha_in == 1 or alpha_out == 1 else \
            nn.Conv2d(in_high_channels,out_high_channels,
                #in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                      kernel_size, 1, padding, dilation, math.ceil(groups - alpha_in * groups), bias)
    def forward(self,x):

        x_h,x_l = x if type(x) is tuple else (x,None)

        x_h = self.downsample(x_h) if self.stride == 2 else x_h # 如果步长为2，则先下采样

        x_h_h = self.convh_h(x_h)
        x_h_l = self.convh_l(self.downsample(x_h)) if self.alpha_out > 0 and not self.is_dw else None
        # 如果输出的低频比例为 0 ，或者是可分离卷积，则这部分为 None

        if x_l is not None:
            x_l_l = self.downsample(x_l) if self.stride == 2 else x_l

            x_l_l = self.convl_l(x_l_l) if self.alpha_out > 0 else None
            if self.is_dw:
                return x_h_h, x_l_l
            else:
                x_l_h = self.convl_h(x_l)
                x_l_h = self.upsample(x_l_h) if self.stride == 1 else x_l_h

                x_h = x_l_h + x_h_h
                x_l = x_h_l + x_l_l if x_h_l is not None and x_l_l is not None else None
                return x_h, x_l
        else:
            return x_h_h, x_h_l

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l

class Conv_BN_ACT(nn.Module):
    def __init__(self, in_high_channels,in_low_channels,out_high_channels,out_low_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_high_channels,in_low_channels,out_high_channels,out_low_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(out_high_channels)
        self.bn_l = None if alpha_out == 0 else norm_layer(out_low_channels)

        self.act_h = activation_layer(inplace=True)
        self.act_l = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act_h(self.bn_h(x_h))
        x_l = self.act_l(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l
