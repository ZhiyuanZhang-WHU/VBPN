import math
import torch
from util.metric_util import standard_psnr_ssim
from model.basic_model.basic_model import BasicModel
import util.util_VPN as util_VPN
from util.util_ELBO import *
import torch.nn.functional as F
import torch.nn as nn


class Model(BasicModel):
    def __init__(self, option, logger, main_dir):
        super().__init__(option, logger, main_dir)

        # added by lhx
        # alpha0
        if self.gpu:
            self.ms_alpha0 = 0.5 * torch.tensor([self.option['train']['var_window']**2], dtype=torch.float32).cuda()
        else:
            self.ms_alpha0 = 0.5 * torch.tensor([self.option['train']['var_window']**2], dtype=torch.float32)

        # kappa0
        if self.gpu:
            self.kappa0 = torch.tensor([self.option['train']['kappa0']], dtype=torch.float32).cuda()
        else:
            self.kappa0 = torch.tensor([self.option['train']['kappa0']], dtype=torch.float32)
        # end added

        self.param_snet = [x for name, x in self.net.named_parameters() if 'snet' in name.lower()]
        self.param_fnet = [x for name, x in self.net.named_parameters() if 'fnet' in name.lower()]
        self.param_knet = [x for name, x in self.net.named_parameters() if 'knet' in name.lower()]
        self.param_rnet = [x for name, x in self.net.named_parameters() if 'rnet' in name.lower()]
        

    # need add: 补充loss计算具体过程，目前没有用到alpha_est
    def __criterion__(self, mu, sigma_est, kinfo_est, alpha_est, 
                      im_hr,
                      ms_lr,
                      sigma_prior,
                      ms_alpha0,
                      kinfo_gt,
                      kappa0,
                      r2,
                      eps2,
                      sf,
                      k_size,
                      penalty_K,
                      shift,
                      downsampler,
                      # PAN部分
                      pan_n,
                      pan_label,
                      pan_sigmaMap,
                      pan_eps2):

        '''
        MS部分loss
        '''
        # KL divergence for Gauss distribution
        if isinstance(mu, list):
            kl_rnet = cal_kl_gauss_simple(mu[0], im_hr, eps2)
            for jj in range(1, len(mu)):
                kl_rnet += cal_kl_gauss_simple(mu[jj], im_hr, eps2)
            kl_rnet /= len(mu)
        else:
            kl_rnet = cal_kl_gauss_simple(mu, im_hr, eps2)

        # KL divergence for Inv-Gamma distribution of the sigma map for noise
        beta0 = sigma_prior * ms_alpha0
        beta = sigma_est * ms_alpha0
        kl_snet = cal_kl_inverse_gamma_simple(beta, ms_alpha0-1, beta0)

        # KL divergence for the kernel
        kl_knet0 = cal_kl_inverse_gamma_simple(kappa0*kinfo_est[:, 0], kappa0-1, kappa0*kinfo_gt[:, 0])
        kl_knet1 = cal_kl_inverse_gamma_simple(kappa0*kinfo_est[:, 1], kappa0-1, kappa0*kinfo_gt[:, 1])
        kl_knet2 = cal_kl_gauss_simple(kinfo_est[:, 2], kinfo_gt[:, 2], r2) * penalty_K[0]
        kl_knet = (kl_knet0 + kl_knet1 + kl_knet2) / 3 * penalty_K[1]

        # reparameter kernel
        k_cov = reparameter_cov_mat(kinfo_est, kappa0, r2)        # resampled covariance matrix, N x 1 x 2 x 2
        kernel = util_VPN.sigma2kernel(k_cov, k_size, sf, shift)        # N x 1 x k x k

        # likelihood
        if isinstance(mu, list):
            lh = cal_likelihood_sisr(ms_lr, kernel, sf, mu[0], eps2, ms_alpha0-1, beta, downsampler)
            for jj in range(1, len(mu)):
                lh += cal_likelihood_sisr(ms_lr, kernel, sf, mu[jj], eps2, ms_alpha0-1, beta, downsampler)
            lh /= len(mu)
        else:
            lh = cal_likelihood_sisr(ms_lr, kernel, sf, mu, eps2, ms_alpha0-1, beta, downsampler)

        loss = lh + kl_rnet + kl_snet + kl_knet

        '''
        PAN部分loss 
        '''
        radius = 3
        out_denoise = mu
        out_denoise = torch.mean(out_denoise, dim=1, keepdim=True)

        # 定义 Sobel 算子
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)   
        
        # 添加维度以匹配图像张量的维度
        if self.gpu:
            sobel_x = sobel_x.view(1, 1, 3, 3).cuda()
            sobel_y = sobel_y.view(1, 1, 3, 3).cuda()
        else:
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)

        # 对图像张量进行 Sobel 算子的卷积操作
        with torch.no_grad():
            out_denoise_gradient_x = F.conv2d(out_denoise, sobel_x, padding=1)
            out_denoise_gradient_y = F.conv2d(out_denoise, sobel_y, padding=1)

            # 计算梯度的幅值
            out_denoise_gradient_magnitude = torch.sqrt(out_denoise_gradient_x**2 + out_denoise_gradient_y**2)

            # pan_label求梯度
            pan_label_gradient_x = F.conv2d(pan_label, sobel_x, padding=1)
            pan_label_gradient_y = F.conv2d(pan_label, sobel_y, padding=1)

            # 计算梯度的幅值
            pan_label_gradient_magnitude = torch.sqrt(pan_label_gradient_x**2 + pan_label_gradient_y**2)

        out_sigma = alpha_est
        log_max = math.log(1e4)
        log_min = math.log(1e-8)
        C = pan_label.shape[1]
        p = 2 * radius + 1
        p2 = p ** 2
        pan_alpha0 = 0.5 * torch.tensor([p2 - 2]).type(pan_sigmaMap.dtype).to(device=pan_sigmaMap.device)
        pan_beta0 = 0.5 * p2 * pan_sigmaMap
        out_denoise_gradient_magnitude[:, C:, ].clamp_(min=log_min, max=log_max)
        err_mean = out_denoise_gradient_magnitude[:, :C, ]
        m2 = torch.exp(out_denoise_gradient_magnitude[:, C:, ])  # variance

        out_sigma.clamp_(min=log_min, max=log_max)
        log_alpha = out_sigma[:, :C, ]
        alpha = torch.exp(log_alpha)
        log_beta = out_sigma[:, C:, ]
        alpha_div_beta = torch.exp(log_alpha - log_beta)

     
        kl_gauss = 0.5 * torch.mean((out_denoise_gradient_magnitude - pan_label_gradient_magnitude) ** 2 / pan_eps2)

        kl_Igamma = torch.mean((alpha - pan_alpha0) * torch.digamma(alpha) + (torch.lgamma(pan_alpha0) - torch.lgamma(alpha))
                               + pan_alpha0 * (log_beta - torch.log(pan_beta0)) + pan_beta0 * alpha_div_beta - alpha)

        pan_lh = 0.5 * math.log(2 * math.pi) + 0.5 * torch.mean(
            (log_beta - torch.digamma(alpha)) + ((out_denoise_gradient_magnitude - pan_label_gradient_magnitude) ** 2 + pan_eps2) * alpha_div_beta)        
        loss = loss + pan_lh + kl_gauss + kl_Igamma


        return loss

    def __feed__(self, data_pair):
        self.optimizer.zero_grad()

        if self.gpu:
         
            ms_lable, ms_lr, ms_blur, kinfo_gt, nlevel, \
                pan_label, pan_n, pan_map, pan_eps2 = [x.cuda() for x in data_pair]
        else:
     
            ms_lable, ms_lr, ms_blur, kinfo_gt, nlevel, \
                pan_label, pan_n, pan_map, pan_eps2 = data_pair


        if self.option['dataset']['train']['ms_add_jpeg']:
            ms_sigma_prior = util_VPN.noise_estimate_fun(ms_lr, ms_blur, self.option['train']['var_window'])
        else:
            ms_sigma_prior = nlevel     # N x 1 x 1 x1 for Gaussian noise
      
        mu, kinfo_est, sigma_est, alpha_est = self.net(ms_lr, pan_n, self.option['dataset']['sf'])
    
        self.loss = self.__criterion__(mu=mu, 
                                       kinfo_est=kinfo_est, 
                                       sigma_est=sigma_est, 
                                       alpha_est=alpha_est, 
                                       im_hr=ms_lable, 
                                       ms_lr=ms_lr,
                                       sigma_prior=ms_sigma_prior,
                                       ms_alpha0=self.ms_alpha0,
                                       kinfo_gt=kinfo_gt,
                                       kappa0=self.kappa0,
                                       r2=self.option['train']['r2'],
                                       eps2=self.option['train']['eps2'],
                                       sf=self.option['dataset']['sf'],
                                       k_size=self.option['dataset']['train']['ms_k_size'],
                                       penalty_K=self.option['train']['penalty_K'],
                                       downsampler=self.option['dataset']['downsampler'],
                                       shift=self.option['dataset']['train']['ms_kernel_shift'],
                                       pan_n=pan_n,
                                       pan_label=pan_label,
                                       pan_sigmaMap=pan_map,
                                       pan_eps2=pan_eps2)

        self.loss.backward()
        

        # clip the gradnorm
        total_norm_S = nn.utils.clip_grad_norm_(self.param_rnet, self.option['train']['optim']['clip_grad_S'])
        total_norm_F = nn.utils.clip_grad_norm_(self.param_snet, self.option['train']['optim']['clip_grad_F'])
        total_norm_K = nn.utils.clip_grad_norm_(self.param_knet, self.option['train']['optim']['clip_grad_K'])
        total_norm_R = nn.utils.clip_grad_norm_(self.param_knet, self.option['train']['optim']['clip_grad_R'])



    def __eval__(self, data_pair):
   
        with torch.no_grad():
            if self.gpu:
                ms_lable, ms_lr, kinfo_gt, \
                    pan_label, pan_n = [x.cuda() for x in data_pair]
            else:
                ms_lable, ms_lr, kinfo_gt, \
                    pan_label, pan_n = data_pair

            mu, kinfo_est, sigma_est, alpha_est = self.net(ms_lr, pan_n, self.option['dataset']['sf'])

        
        psnr, ssim = standard_psnr_ssim(mu, ms_lable, mode=self.option['train']['metric_mode'])
        return psnr, ssim
