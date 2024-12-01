# -*- coding: utf-8 -*-
# @Time    : 3/5/23 3:46 PM
# @File    : metric_util.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import cv2 as cv
import numpy as np
import torch.nn as nn
from skimage import img_as_ubyte
from torchvision.models import vgg19
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import imgvision as iv
from numpy.linalg import norm

def image_mean(image):
        mean = np.mean(image)
        return mean

def image_var(image, mean):
        m, n = np.shape(image)
        var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
        return var

def images_cov(image1, image2, mean1, mean2):
        m, n = np.shape(image1)
        cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
        return cov

def UQI(O, F):
        '''
        :param O: 原始图像
        :param F: 滤波后的图像
        '''
        meanO = image_mean(O)
        meanF = image_mean(F)
        varO = image_var(O, meanO)
        varF = image_var(F, meanF)
        covOF = images_cov(O, F, meanO, meanF)
        UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
        return UQI

def SAM(x_true, x_pred):
    """
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    """

    assert x_true.ndim == 3 and x_true.shape == x_pred.shape
    dot_sum = np.sum(x_true * x_pred, axis=2)
    norm_true = norm(x_true, axis=2)
    norm_pred = norm(x_pred, axis=2)

    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0

    sam = np.mean(res)
    return sam


def standard_psnr_ssim(input, target, mode='y'):
    """
    psnr and ssim
    @param input: tensor (1, c, h, w)
    @param target: tensor (1, c, h, w)
    @param mode: ['gray', 'rgb', 'ycbcy']
    @return: psnr value and ssim value
    """
    input = input.data.cpu().numpy().clip(0, 1)
    target = target.data.cpu().numpy().clip(0, 1)
    input = np.transpose(input[0, :], (1, 2, 0))
    target = np.transpose(target[0, :], (1, 2, 0))
    input, target = img_as_ubyte(input), img_as_ubyte(target)
    if mode == 'y':
        input = cv.cvtColor(input, cv.COLOR_RGB2YCrCb)[:, :, :1]
        target = cv.cvtColor(target, cv.COLOR_RGB2YCrCb)[:, :, :1]

    psnr = peak_signal_noise_ratio(image_true=target, image_test=input)
    ssim = structural_similarity(im1=input, im2=target, channel_axis=2)
    return psnr, ssim

def imgvision_psnr_ssim(input, target):
    input = input.data.cpu().numpy().clip(0, 1)
    target = target.data.cpu().numpy().clip(0, 1)
    input = np.transpose(input[0, :], (1, 2, 0))
    target = np.transpose(target[0, :], (1, 2, 0))
    #input, target = img_as_ubyte(input), img_as_ubyte(target)
    Metric = iv.spectra_metric(target,input)
    sam = SAM(x_true = target, x_pred = input)
    #评价PSNR：
    # PSNR = Metric.PSNR()
    PSNR = peak_signal_noise_ratio(image_true=target, image_test=input)
    #评价SSIM：
    # SSIM = Metric.SSIM() 
    SSIM = structural_similarity(im1=input, im2=target, channel_axis=2)
    #评价ERGAS:
    ERGAS = Metric.ERGAS()
    Q = 0.0

    for i in range(input.shape[2]):
        Q += UQI(target[:,:,i],input[:,:,i])
    Q = Q/4.0

    #评价RMSE:    
    d = (target - input) ** 2
    RMSE = np.sqrt(np.sum(d) / (d.shape[0] * d.shape[1]))


    return PSNR, SSIM, sam, ERGAS, Q, RMSE

def Q(a, b):
    a = a.reshape(a.shape[0] * a.shape[1])
    b = b.reshape(b.shape[0] * b.shape[1])
    temp = np.cov(a, b)
    d1 = temp[0, 0]
    cov = temp[0, 1]
    d2 = temp[1, 1]
    m1 = np.mean(a)
    m2 = np.mean(b)
    Q = 4 * cov * m1 * m2 / (d1 + d2) / (m1 ** 2 + m2 ** 2)

    return Q

def D_lamda(ps, l_ms):
    ps = ps.data.cpu().numpy().clip(0, 1)
    l_ms = l_ms.data.cpu().numpy().clip(0, 1)
    ps = np.transpose(ps[0, :], (1, 2, 0))
    l_ms = np.transpose(l_ms[0, :], (1, 2, 0))
    ps, l_ms = img_as_ubyte(ps), img_as_ubyte(l_ms)
    L = ps.shape[2]
    sum = 0.0
    for i in range(L):
        for j in range(L):
            if j != i:
                # print(np.abs(Q(ps[:, :, i], ms[:, :, j]) - Q(l_ps[:, :, i], l_ms[:, :, j])))
                sum += np.abs(Q(ps[:, :, i], ps[:, :, j]) - Q(l_ms[:, :, i], l_ms[:, :, j]))
    return sum / L / (L - 1)


def D_s(ps, l_ms, pan):
    ps = ps.data.cpu().numpy().clip(0, 1)
    l_ms = l_ms.data.cpu().numpy().clip(0, 1)
    pan = pan.data.cpu().numpy().clip(0, 1)
    ps = np.transpose(ps[0, :], (1, 2, 0))
    l_ms = np.transpose(l_ms[0, :], (1, 2, 0))
    pan = np.transpose(pan[0, :], (1, 2, 0))
    ps, l_ms, pan = img_as_ubyte(ps), img_as_ubyte(l_ms), img_as_ubyte(pan)
    L = ps.shape[2]
    # h, w = pan.shape
    # l_pan = cv2.resize(pan, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    l_pan = cv.pyrDown(pan)
    l_pan = cv.pyrDown(l_pan)
    sum = 0.0
    for i in range(L):
        sum += np.abs(Q(ps[:, :, i], pan) - Q(l_ms[:, :, i], l_pan))
    return sum / L
