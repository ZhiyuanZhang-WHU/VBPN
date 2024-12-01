# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:26 PM
# @File    : channel.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import torch.nn as nn


class Contrast_Channel_Attnetion(nn.Module):
    def __init__(self, in_ch, reduction):
        super(Contrast_Channel_Attnetion, self).__init__()
        self.contrast = self.stdv_chaneels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch // reduction, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_ch // reduction, out_channels=in_ch, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        out = x * y
        return out

    def mean_channels(self, F):
        assert (F.dim() == 4)
        spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
        mean = spatial_sum / (F.size(2) * F.size(3))
        return mean

    def stdv_chaneels(self, F):
        assert (F.dim() == 4)
        mean = self.mean_channels(F)
        variance = (F - mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
        return variance.pow(0.5)
