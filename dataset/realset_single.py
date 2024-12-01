import sys
import cv2  as cv
import numpy as np
import torch
import random

import util.data_util as data_util
from util import util_VPN
from dataset.basic_dataset.basic_unpair import BasicDataSetUnPair
from scipy.io import loadmat
from util.ResizeRight.resize_right import resize
from osgeo import gdal

def normalfunc(image):
    image = np.float32(image)
    bands = image.shape[2]
    re_image = np.zeros(image.shape)
    for i in range(bands):
        msi_slice = image[:, :, i]
        min_patch = np.min(msi_slice[np.nonzero(msi_slice)])
        max_patch = np.max(msi_slice, axis=(0, 1))
        re_image[:, :, i] = np.float32(msi_slice - min_patch) / (max_patch - min_patch)
        re_image[:, :, i][re_image[:, :, i]<0]=0

    return re_image


class TestDataSet(BasicDataSetUnPair):
    def __init__(self, input_dir = "/home/zzy/ZW_pansharpening/data1", ms_patch_size=64, sf=4, 
                 ms_k_size=21, 
                 ms_kernel_shift=False, 
                 downsampler='bicubic', 
                 # PAN部分
                #  pan_patch_size=256,
                 pan_clip=False,
                 mspath = '/home/Public/Data/QB/QB_3_MUL.TIF',
                 panpath = '/home/Public/Data/QB/QB_3_PAN.TIF'):
        super(TestDataSet, self).__init__(input_dir, mode='gray')

        self.sf = sf
        self.downsampler = downsampler
        '''
        MS部分
        '''
        self.ms_hr_size = ms_patch_size
        self.ms_k_size = ms_k_size
        self.ms_kernel_shift = ms_kernel_shift


        # # generate fixed Gaussian noise
        # self.fixed_noise = self.generate_noise()

        '''
        PAN部分
        '''
        self.pan_clip = pan_clip
        self.mspath = mspath
        self.panpath = panpath
 
        # self.clip = clip
        # self.levels = levels
        # if task.upper() == 'poisson'.upper():
        #     self.add_noise = data_util.add_poisson_noise
        # elif task.upper() == 'gaussian'.upper():
        #     self.add_noise = data_util.add_gaussian_noise

    # need add: 对准输出的变量
    def __getitem__(self):
        ms_label, pan_label = self.__get_image__()

        '''
        处理ms_label, 参考vir
        '''  
        # blur kernel
        kernel, kernel_infos = util_VPN.shifted_anisotropic_Gaussian(k_size=self.ms_k_size,
                                                                      sf=self.sf,
                                                                      lambda_1=1.6**2,
                                                                      lambda_2=1.6**2,
                                                                      theta=0,
                                                                      shift=self.ms_kernel_shift)

        # blurring
        ms_blur = util_VPN.imconv_np(ms_label, kernel, padding_mode='reflect', correlate=False)
        ms_blur = np.clip(ms_blur, a_min=0.0, a_max=1.0)

        # downsampling
        if self.downsampler.lower() == 'direct':
            ms_blur = ms_blur[::self.sf, ::self.sf,]
        elif self.downsampler.lower() == 'bicubic':
            ms_blur = resize(ms_blur, scale_factors=1/self.sf).astype(np.float32)
        else:
            sys.exit('Please input corrected downsampler: Direct or Bicubic')

        # need add: 这里与virnet稍有区别
        # adding noise
        # c x h x w
        ms_lr = torch.from_numpy(ms_blur.transpose([2,0,1])).type(torch.float32) 
        ms_label = torch.from_numpy(ms_label.transpose([2,0,1])).type(torch.float32)      # c x h x w            # 3


        '''
        处理PAN_lable, 参考vdn
        '''

        # downsampling
        if self.downsampler.lower() == 'direct':
            pan_n = pan_label[::self.sf, ::self.sf,]
        elif self.downsampler.lower() == 'bicubic':
            pan_label_l = resize(pan_label, scale_factors=1/self.sf).astype(np.float32)
            pan_n = pan_label_l
        else:
            sys.exit('Please input corrected downsampler: Direct or Bicubic')

        pan_label = pan_label.astype(np.float32)

        tensor_pan_label = data_util.image2tensor(pan_label)
        tensor_pan_n = data_util.image2tensor(pan_n)
        self.gt = ms_label
        self.lrms = ms_lr
        self.pan = tensor_pan_n
        self.pan_label = tensor_pan_label



        # return ms_label, ms_lr, ms_blur, kernel_infos, nlevel, \
        #     tensor_pan_label, tensor_pan_n, tensor_map, tensor_eps2
        return self.gt, self.pan_label



        # return ms_label, ms_lr, kernel_infos, \
        #     tensor_pan_label, tensor_pan_n

        # level = np.random.choice(self.levels)
        # gt = self.__get_image__(item)
        # noisy = self.add_noise(gt.copy(), level=level, clip=self.clip)
        # tensor_g = data_util.image2tensor(gt)
        # tensor_n = data_util.image2tensor(noisy)
        # return tensor_n, tensor_g
    
    # need add: 做精细化修正
    def __get_image__(self):
        # img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)
        # rawMSI = gdal.Open('/home/Public/Data/QB/QB_3_MUL.TIF').ReadAsArray()
        # rawPAN = gdal.Open('/home/Public/Data/QB/QB_3_PAN.TIF').ReadAsArray()

        rawMSI = gdal.Open(self.mspath).ReadAsArray()
        rawPAN = gdal.Open(self.panpath).ReadAsArray()

        # print("rawMSI:",rawMSI.shape, "rawPAN:",rawPAN.shape)
        rawMSI = rawMSI.transpose(1,2,0)
        rawPAN = np.expand_dims(rawPAN, axis= 2)
        ms_label = normalfunc(rawMSI)
        pan_label = normalfunc(rawPAN)

        return ms_label, pan_label     

    # def generate_noise(self):
    #     h_max, w_max = 1, 1
    #     for im_path in self.hr_path_list:
    #         im = util_image.imread(im_path, chn='bgr', dtype='uint8')
    #         h, w = im.shape[:2]
    #         if h_max < h: h_max = h
    #         if w_max < w: w_max = w
    #     h_down, w_down = math.ceil(h_max / self.sf), math.ceil(w_max / self.sf)

    #     g =torch.Generator()
    #     g.manual_seed(self.seed)
    #     noise = torch.randn([h_down, w_down, 3], generator=g, dtype=torch.float32).numpy()
    #     return noise