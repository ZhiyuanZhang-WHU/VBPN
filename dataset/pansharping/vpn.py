
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

def sigma_estimate(im_noisy, im_gt, win, sigma_spatial):
    noise2 = (im_noisy - im_gt) ** 2
    sigma2_map_est = cv.GaussianBlur(noise2, (win, win), sigma_spatial)
    sigma2_map_est = sigma2_map_est.astype(np.float32)
    sigma2_map_est = np.where(sigma2_map_est<1e-10, 1e-10, sigma2_map_est)
    if sigma2_map_est.ndim == 2:
        sigma2_map_est = sigma2_map_est[:, :, np.newaxis]
    return sigma2_map_est

class TrainDataSet(BasicDataSetUnPair):
    def __init__(self, input_dir, ms_patch_size=64, sf=4, 
                 ms_k_size=21, 
                 ms_kernel_shift=False, 
                 downsampler='bicubic', 
                 ms_noise_level=[0.1, 15], 
                 ms_noise_jpeg=[0.1, 10], 
                 ms_add_jpeg=False,
                 # PAN部分
                #  pan_patch_size=256,
                 pan_clip=False,
                 pan_nlevels=70,
                 pan_task='gaussian'
                 ):
    # def __init__(self, input_dir, patch_size, levels, task='poisson', mode='gray', clip=False):
        super(TrainDataSet, self).__init__(input_dir=input_dir, mode='gray')

        self.sf = sf
        self.downsampler = downsampler
        '''
        MS部分
        '''
        self.ms_hr_size = ms_patch_size
        self.ms_k_size = ms_k_size
        self.ms_kernel_shift = ms_kernel_shift

        self.ms_noise_types = ['Gaussian',]
        if ms_add_jpeg:
            self.ms_noise_types.append('JPEG')  

        # noise level for Gaussian noise
        assert ms_noise_level[0] < ms_noise_level[1]
        self.ms_noise_level = ms_noise_level
        assert ms_noise_jpeg[0] < ms_noise_jpeg[1]
        self.ms_noise_jpeg = ms_noise_jpeg

        '''
        PAN部分
        '''
        self.pan_clip = pan_clip
        self.pan_nlevels = pan_nlevels
        # self.patch_size = pan_patch_size
        if pan_task.upper() == 'poisson'.upper():
            self.pan_add_noise = data_util.add_poisson_noise
        elif pan_task.upper() == 'gaussian'.upper():
            self.pan_add_noise = data_util.add_gaussian_noise
        self.pan_eps2 = 1e-5
        self.pan_sigma_spatial = 3
        self.pan_window_size = 2 * self.pan_sigma_spatial + 1


    def __getitem__(self, item):
        ms_label, pan_label = self.__get_image__(item)

        '''
        处理ms_label
        '''  
        # blur kernel
        lam1 = random.uniform(0.2, self.sf)
        lam2 = random.uniform(lam1, self.sf) if random.random() < 0.7 else lam1
        theta = random.uniform(0, np.pi)
        kernel, kernel_infos = util_VPN.shifted_anisotropic_Gaussian(k_size=self.ms_k_size,
                                                                      sf=self.sf,
                                                                      lambda_1=lam1**2,
                                                                      lambda_2=lam2**2,
                                                                      theta=theta,
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

        # adding noise
        noise_type = random.sample(self.ms_noise_types, k=1)[0]
        if noise_type == 'Gaussian':
            std = random.uniform(self.ms_noise_level[0], self.ms_noise_level[1]) / 255.0
            ms_lr = ms_blur + torch.randn(ms_blur.shape, dtype=torch.float32).numpy() * std
            ms_lr = np.clip(ms_lr, a_min=0, a_max=1.0)
        elif noise_type == 'JPEG':
            qf = self.random_qf()
            std = random.uniform(self.ms_noise_jpeg[0], self.ms_noise_jpeg[1]) / 255.0
            im_noisy = ms_blur + torch.randn(ms_blur.shape, dtype=torch.float32).numpy() * std
            im_noisy = np.clip(im_noisy, a_min=0.0, a_max=1.0)
            ms_lr = util_VPN.jpeg_compress(im_noisy, int(qf), chn_in='rgb')
        else:
            sys.exit('Please input corrected noise type: JPEG or Gaussian')

        ms_label = torch.from_numpy(ms_label.transpose([2,0,1])).type(torch.float32)        # c x h x w
        ms_lr = torch.from_numpy(ms_lr.transpose([2,0,1])).type(torch.float32)        # c x h x w
        ms_blur = torch.from_numpy(ms_blur.transpose([2,0,1])).type(torch.float32)    # c x h x w
        kernel_infos = torch.from_numpy(kernel_infos).type(torch.float32)             # 3
        nlevel = torch.tensor([std], dtype=torch.float32).unsqueeze(-1).unsqueeze(-1) # 1 x 1 x 1


        '''
        处理PAN_lable
        '''
        pan_nlevel = np.random.choice(self.pan_nlevels)

        # downsampling
        if self.downsampler.lower() == 'direct':
            pan_label_down = pan_label[::self.sf, ::self.sf,]
            pan_n = pan_label_down
        elif self.downsampler.lower() == 'bicubic':
            pan_label_down = resize(pan_label, scale_factors=1/self.sf).astype(np.float32)
            pan_n = pan_label_down
        else:
            sys.exit('Please input corrected downsampler: Direct or Bicubic')

       

        pan_n = self.pan_add_noise(pan_n.copy(), level=pan_nlevel, clip=self.pan_clip)

        sigma2_map_est = sigma_estimate(im_noisy=pan_n, im_gt=pan_label_down, win=self.pan_window_size,
                                        sigma_spatial=self.pan_sigma_spatial)
        tensor_pan_label = data_util.image2tensor(pan_label_down)
        tensor_pan_n = data_util.image2tensor(pan_n)
        tensor_map = data_util.image2tensor(sigma2_map_est)
        tensor_eps2 = torch.tensor([self.pan_eps2], dtype=torch.float32).reshape((1, 1, ))


        return ms_label, ms_lr, ms_blur, kernel_infos, nlevel, \
            tensor_pan_label, tensor_pan_n, tensor_map, tensor_eps2
    
    def __get_image__(self, item):
        # img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)

        data = loadmat(self.input_paths[item])
        # pan = data['pan'].astype(np.float32)
        # ms=data['ms'].astype(np.float32)#[H,w,C]
        # ms_up = data['ms_up'].astype(np.float32)#[H,w,C]
        pan_label = data['pan_label'].astype(np.float32)
        ms_label = data['ms_label'].astype(np.float32)#[H,w,C]

        return ms_label, pan_label        

    def random_qf(self):
        start = list(range(30, 50, 5)) + [60, 70, 80]
        end = list(range(35, 50, 5)) + [60, 70, 80, 95]
        ind_range = random.randint(0, len(start)-1)
        qf = random.randint(start[ind_range], end[ind_range])
        return qf

# need add: Dataset test部分
class TestDataSet(BasicDataSetUnPair):
    def __init__(self, input_dir, ms_patch_size=64, sf=4, 
                 ms_k_size=21, 
                 ms_kernel_shift=False, 
                 downsampler='bicubic', 
                 ms_noise_type='Gaussian',
                 # PAN部分
                #  pan_patch_size=256,
                 pan_clip=False,
                 pan_nlevels=70,
                 pan_task='gaussian'):
        super(TestDataSet, self).__init__(input_dir, mode='gray')

        self.sf = sf
        self.downsampler = downsampler
        '''
        MS部分
        '''
        self.ms_hr_size = ms_patch_size
        self.ms_k_size = ms_k_size
        self.ms_kernel_shift = ms_kernel_shift

        self.ms_noise_type = ms_noise_type


        '''
        PAN部分
        '''
        self.pan_clip = pan_clip
        self.pan_nlevels = pan_nlevels
        if pan_task.upper() == 'poisson'.upper():
            self.pan_add_noise = data_util.add_poisson_noise
        elif pan_task.upper() == 'gaussian'.upper():
            self.pan_add_noise = data_util.add_gaussian_noise
        self.pan_eps2 = 1e-6
        self.pan_sigma_spatial = 3
        self.pan_window_size = 2 * self.pan_sigma_spatial + 1

    def __getitem__(self, item):
        ms_label, pan_label = self.__get_image__(item)

        '''
        处理ms_label
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


        noise_type = self.ms_noise_type
        if noise_type == 'Gaussian':
            std = 2.55 / 255.0
            ms_lr = self.pan_add_noise(ms_blur, level= 2.55, clip= False)
            ms_lr = np.clip(ms_lr, a_min=0, a_max=1.0)
        elif noise_type == 'JPEG':
            qf = 40
            std = 2.55 / 255.0
            im_noisy = ms_blur + torch.randn(ms_blur.shape, dtype=torch.float32).numpy() * std
            im_noisy = np.clip(im_noisy, a_min=0.0, a_max=1.0)
            ms_lr = util_VPN.jpeg_compress(im_noisy, int(qf), chn_in='rgb')
        else:
            sys.exit('Please input corrected noise type: JPEG or Gaussian')

        ms_label = torch.from_numpy(ms_label.transpose([2,0,1])).type(torch.float32)        # c x h x w
        ms_lr = torch.from_numpy(ms_lr.transpose([2,0,1])).type(torch.float32)        # c x h x w
        kernel_infos = torch.from_numpy(kernel_infos).type(torch.float32)             # 3


        '''
        处理PAN_lable
        '''
        pan_nlevel = np.random.choice(self.pan_nlevels)

        # downsampling
        if self.downsampler.lower() == 'direct':
            pan_label_down = pan_label[::self.sf, ::self.sf,]
        elif self.downsampler.lower() == 'bicubic':
            pan_label_down = resize(pan_label, scale_factors=1/self.sf).astype(np.float32)
            pan_n = pan_label_down
        else:
            sys.exit('Please input corrected downsampler: Direct or Bicubic')

        pan_n = self.pan_add_noise(pan_n.copy(), level=pan_nlevel, clip=self.pan_clip)
        tensor_pan_label = data_util.image2tensor(pan_label)
        tensor_pan_n = data_util.image2tensor(pan_n)



        return ms_label, ms_lr, kernel_infos, \
            tensor_pan_label, tensor_pan_n



    
    # need add: 做精细化修正
    def __get_image__(self, item):
        
        data = loadmat(self.input_paths[item])
  
        pan_label = data['pan'].astype(np.float32)
        ms_label = data['ms'].astype(np.float32)#[H,w,C]

        return ms_label, pan_label   

