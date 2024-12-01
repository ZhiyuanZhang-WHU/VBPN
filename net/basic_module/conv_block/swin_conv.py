# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:29 PM
# @File    : swin_conv.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange
from net.basic_module.transformer.swin_transformer import SwinTransformerBlock, WMSA
from net.basic_module.attention.spatial import Enhance_Spatial_Attention


class SwinConvTransBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(SwinConvTransBlock, self).__init__()
        self.conv_dim = input_dim // 2
        self.trans_dim = input_dim // 2
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'

        self.trans_block = SwinTransformerBlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size,
                                                self.drop_path, self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, output_dim, 1, 1, 0, bias=True)

        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class ESWTConvBlock(nn.Module):
    """Enhance Shiffed Window Transformer Conv Block
    by Liang H. ![](https://qiniu.lianghao.work/202302140938970.png)
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='SW', input_resolution=None):
        super(ESWTConvBlock, self).__init__()
        self.input_dim = input_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if input_resolution <= window_size:
            type = 'W'

        """swin-transformer area with esa"""
        self.sw_ln_1 = nn.LayerNorm(input_dim // 2)
        self.sw_msa = WMSA(input_dim // 2, input_dim // 2, head_dim, window_size, type)
        self.sw_ln_2 = nn.LayerNorm(input_dim // 2)
        self.sw_mlp = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim // 2)
        )
        self.sw_esa = Enhance_Spatial_Attention(in_ch=input_dim // 2)

        """convolution enhance locality"""
        self.res_conv = nn.Sequential(
            nn.Conv2d(input_dim // 2, input_dim // 2, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, input_dim // 2, 3, 1, 1, bias=False)
        )
        self.conv_1x1 = nn.Conv2d(input_dim, output_dim, 1, 1, 0, bias=True)

    def forward(self, input):
        conv_x, trans_x = torch.split(input, (self.input_dim // 2, self.input_dim // 2), dim=1)
        # conv_x = self.res_conv(conv_x) + conv_x
        conv_x = self.res_conv(conv_x)
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.drop_path(self.sw_msa(self.sw_ln_1(trans_x))) + trans_x
        trans_x = self.drop_path(self.sw_mlp(self.sw_ln_2(trans_x))) + trans_x
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.sw_esa(trans_x)
        concate = torch.cat([conv_x, trans_x], dim=1)
        output = self.conv_1x1(concate) + input
        return output


class ESWTConvBlockB(nn.Module):
    """add : resconv """
    """Enhance Shiffed Window Transformer Conv Block
    by Liang H. ![](https://qiniu.lianghao.work/202302140938970.png)
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='SW', input_resolution=None):
        super(ESWTConvBlockB, self).__init__()
        self.input_dim = input_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if input_resolution <= window_size:
            type = 'W'

        """swin-transformer area with esa"""
        self.sw_ln_1 = nn.LayerNorm(input_dim // 2)
        self.sw_msa = WMSA(input_dim // 2, input_dim // 2, head_dim, window_size, type)
        self.sw_ln_2 = nn.LayerNorm(input_dim // 2)
        self.sw_mlp = nn.Sequential(
            nn.Linear(input_dim // 2, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, input_dim // 2)
        )
        self.sw_esa = Enhance_Spatial_Attention(in_ch=input_dim // 2)

        """convolution enhance locality"""
        self.res_conv = nn.Sequential(
            nn.Conv2d(input_dim // 2, input_dim // 2, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, input_dim // 2, 3, 1, 1, bias=False)
        )
        self.conv_1x1 = nn.Conv2d(input_dim, output_dim, 1, 1, 0, bias=True)

    def forward(self, input):
        conv_x, trans_x = torch.split(input, (self.input_dim // 2, self.input_dim // 2), dim=1)
        conv_x = self.res_conv(conv_x) + conv_x
        # conv_x = self.res_conv(conv_x)
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.drop_path(self.sw_msa(self.sw_ln_1(trans_x))) + trans_x
        trans_x = self.drop_path(self.sw_mlp(self.sw_ln_2(trans_x))) + trans_x
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        trans_x = self.sw_esa(trans_x)
        concate = torch.cat([conv_x, trans_x], dim=1)
        output = self.conv_1x1(concate) + input
        return output

class SwinTBlock(nn.Module):
    """mlp - ESA"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='SW', input_resolution=None):
        super(SwinTBlock, self).__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if input_resolution <= window_size:
            type = 'W'

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, type)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, output_dim)
        )
        self.dw_conv = Enhance_Spatial_Attention(in_ch=input_dim)

    def forward(self, input):
        x0 = Rearrange('b c h w -> b h w c')(input)
        x1 = self.drop_path(self.msa(self.layer_norm_1(x0)))
        x1 = x1 + x0
        x2 = self.drop_path(self.mlp(self.layer_norm_2(x1))) + x1
        output = Rearrange('b h w c -> b c h w')(x2)
        output = self.dw_conv(output)
        return output


class ESWTConvBlock_B(nn.Module):
    """new conv architecture"""

    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='SW', input_resolution=None):
        super(ESWTConvBlock_B, self).__init__()
        self.input_dim = input_dim
        self.conv_1 = nn.Conv2d(input_dim, input_dim, 1, stride=1, padding=0, bias=True)
        self.swin_block = SwinTBlock(input_dim // 2, output_dim // 2, head_dim, window_size, drop_path, type=type,
                                     input_resolution=input_resolution)
        self.resconv_block = nn.Sequential(
            nn.Conv2d(input_dim // 2, input_dim // 2, 3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, input_dim // 2, 3, stride=1, padding=1, bias=False)
        )
        self.conv_2 = nn.Conv2d(input_dim, input_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        # x1 = self.conv_1(input)
        x1 = input
        conv_x, trans_x = torch.split(x1, (self.input_dim // 2, self.input_dim // 2), dim=1)
        conv_x = self.resconv_block(conv_x)
        trans_x = self.swin_block(trans_x)
        x2 = torch.cat([conv_x, trans_x], dim=1)
        output = self.conv_2(x2) + x1
        return output