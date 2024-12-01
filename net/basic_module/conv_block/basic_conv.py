# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:27 PM
# @File    : basic_conv.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import torch.nn as nn
from net.basic_module.attention.spatial_attention import Enhance_Spatial_Attention


class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        f1 = self.conv(x)
        return f1


class Conv_X_1(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super(Conv_X_1, self).__init__()
        self.conv_x = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=1,
                                  stride=1, padding=0, bias=bias)

    def forward(self, x):
        f1 = self.relu((self.conv_x(x)))
        f2 = self.conv_1x1(f1)
        return f2


class ExtractConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super(ExtractConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1x1 = nn.Conv2d(out_ch * 2, out_ch, 1, 1, 0, bias=False)
    def forward(self, input):
        f1 = self.relu(self.conv(input))
        f2 = self.conv_1x1(f1)
        return f2


class SpachConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super(SpachConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        f1 = input + self.conv(f1)
        output = self.relu(f1)
        return output


class ResConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super(ResConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, stride, padding, bias=bias)
        )

    def forward(self, x):
        output = self.conv(x) + x
        return output


class ESAConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super(ESAConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, bias=bias),
            nn.GELU(),
            nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, bias=bias)
        )
        self.gelu = nn.GELU()
        self.conv_1x1 = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.esa = Enhance_Spatial_Attention(in_ch=out_ch)

    def forward(self, input):
        x1 = self.conv(input) + input
        x2 = self.conv_1x1(self.gelu(x1))
        output = self.esa(x2)
        return output


class Conv_1_X(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super(Conv_1_X, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // 2, kernel_size=1,
                                  stride=1, padding=0, bias=True)
        self.conv_x = nn.Conv2d(in_channels=in_ch // 2, out_channels=out_ch, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f1 = self.relu(self.conv_1x1(x))
        f2 = self.conv_x(f1)
        return f2


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, kernel_size, stride, padding, dilation, bias):
        super(DepthWiseConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=in_ch, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out


class PointWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, bias):
        super(PointWiseConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = self.conv(x)
        return out


class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, bias=True):
        super(DepthWiseSeparableConv, self).__init__()
        self.depthwise = DepthWiseConv(in_ch=in_ch, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, bias=bias)

        self.pointwise = PointWiseConv(in_ch=in_ch, out_ch=out_ch, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        out = self.pointwise(x)
        return out


class SubPixelConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True, upscale=2, conv=BasicConv):
        super(SubPixelConv, self).__init__()
        self.conv = conv(in_ch=in_ch, out_ch=out_ch * (upscale ** 2),
                         kernel_size=kernel_size, padding=padding, stride=stride,
                         bias=bias)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=upscale)

    def forward(self, x):
        out = self.pixelshuffle(self.conv(x))
        return out


class DownSample_UnShuffleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True, downscale=2):
        super().__init__()
        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor=downscale)
        self.conv = nn.Conv2d(in_channels=in_ch * downscale * downscale, out_channels=out_ch,
                              kernel_size=kernel_size, padding=padding, stride=stride,
                              bias=bias)
        # self.conv = conv(in_ch=in_ch * downscale * downscale, out_ch=out_ch, kernel_size=3, padding=padding,
        #                  stride=stride, bias=bias)

    def forward(self, x):
        x = self.pixelunshuffle(x)
        out = self.conv(x)
        return out


class DownSample_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True, downscale=2):
        super(DownSample_Conv, self).__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.down(x)


class UpSample_Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, bias=True, upscale=2):
        super(UpSample_Conv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    import torch
    from thop import profile

    # conv = DepthWiseSeparableConv(in_ch=3, out_ch=3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
    conv = DownSample_UnShuffleConv(3, 3, 3, stride=1, padding=1, bias=True, downscale=2, conv=Conv_1_X)
    a = torch.randn(7, 3, 100, 100)
    b = conv(a)
    print(b.shape)
    flops, params = profile(conv, inputs=(a,), verbose=False)
    print(f"flops = {flops / 1e6}M")
    print(f"params = {params / 1e3}K")