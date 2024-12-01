# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:26 PM
# @File    : spatial.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import torch.nn as nn
import torch.nn.functional as F


class Enhance_Spatial_Attention(nn.Module):
    def __init__(self, in_ch):
        num_feature = in_ch
        f = in_ch // 4
        super(Enhance_Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_feature, out_channels=f, kernel_size=1)
        self.conv_f = nn.Conv2d(in_channels=f, out_channels=f, kernel_size=1)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = nn.Conv2d(f, f, 3, 1, 1)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = nn.Conv2d(f, f, 3, 1, 1)
        self.conv3_ = nn.Conv2d(f, f, 3, 1, 1)
        self.conv4 = nn.Conv2d(f, in_ch, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        out = x * m
        return out