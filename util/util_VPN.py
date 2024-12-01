import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import math
from scipy.special import softmax
import scipy.ndimage as snd
from skimage import img_as_ubyte, img_as_float32, img_as_float64


from util.ResizeRight.resize_right import resize

def getGaussianKernel2DCenter(H, W, center, scale):
    '''
    Generating Gaussian kernel (H x W) with std=scale.

    '''
    centerH = center[0]
    centerW = center[1]
    ii, jj = [x.astype(np.float64) for x in np.meshgrid(np.arange(H), np.arange(W), indexing='ij')]
    kk = np.exp( (-(ii-centerH)**2-(jj-centerW)**2) / (2*scale**2) )
    kk /= kk.sum()
    return kk


def inverse_gamma_kernel(ksize, chn):
    '''
    Create the gauss kernel for inverge gamma prior.
    out:
        kernel: chn x 1 x k x k
    '''
    scale = 0.3 * ((ksize-1)*0.5 -1) + 0.8  # opencv setting
    kernel = getGaussianKernel2D(ksize, sigma=scale)
    kernel = np.tile(kernel[np.newaxis, np.newaxis,], [chn, 1, 1, 1])
    kernel = torch.from_numpy(kernel).type(torch.float32)
    return kernel


def getGaussianKernel2D(ksize, sigma=-1):
    kernel1D = cv2.getGaussianKernel(ksize, sigma)
    kernel2D = np.matmul(kernel1D, kernel1D.T)
    ZZ = kernel2D / kernel2D.sum()
    return ZZ


def conv_multi_chn(x, kernel):
    '''
    In:
        x: B x chn x h x w, tensor
        kernel: chn x 1 x k x k, tensor
    '''
    x_pad = F.pad(x, pad=[kernel.shape[-1]//2, ]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=x.shape[1])

    return y


def noise_estimate_fun(im_noisy, im_gt, k_size):
    '''
    Estatmate the variance map.
    Input:
        im_noisy: N x c x h x w
    '''
    kernel = inverse_gamma_kernel(k_size, im_noisy.shape[1]).to(im_noisy.device)
    err2 = (im_noisy - im_gt) ** 2
    sigma_prior = conv_multi_chn(err2, kernel)
    sigma_prior.clamp_(min=1e-10)
    return sigma_prior

def sigma2kernel(sigma, k_size=21, sf=3, shift=False):
    '''
    Generate Gaussian kernel according to cholesky decomposion.
    Input:
        sigma: N x 1 x 2 x 2 torch tensor, covariance matrix
        k_size: integer, kernel size
        sf: scale factor
    Output:
        kernel: N x 1 x k x k torch tensor
    '''
    try:
        sigma_inv = torch.inverse(sigma)
    except:
        sigma_disturb = sigma + torch.eye(2, dtype=sigma.dtype, device=sigma.device).unsqueeze(0).unsqueeze(0) * 1e-5
        sigma_inv = torch.inverse(sigma_disturb)

    # Set expectation position (shifting kernel for aligned image)
    if shift:
        center = k_size // 2 + 0.5 * (sf - k_size % 2)                         # + 0.5 * (sf - k_size % 2)
    else:
        center = k_size // 2

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(k_size), torch.arange(k_size))
    Z = torch.stack((X, Y), dim=2).to(device=sigma.device, dtype=sigma.dtype).view(1, -1, 2, 1)      # 1 x k^2 x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - center                                                        # 1 x k^2 x 2 x 1
    ZZ_t = ZZ.permute(0, 1, 3, 2)                                          # 1 x k^2 x 1 x 2
    ZZZ = -0.5 * ZZ_t.matmul(sigma_inv).matmul(ZZ).squeeze(-1).squeeze(-1) # N x k^2
    kernel = F.softmax(ZZZ, dim=1)                                         # N x k^2

    return kernel.view(-1, 1, k_size, k_size)                # N x 1 x k x k

def conv_multi_kernel_tensor(im_hr, kernel, sf, downsampler):
    '''
    Degradation model by Pytorch.
    Input:
        im_hr: N x c x h x w
        kernel: N x 1 x k x k
        sf: scale factor
    '''
    im_hr_pad = F.pad(im_hr, (kernel.shape[-1] // 2,)*4, mode='reflect')
    im_blur = F.conv3d(im_hr_pad.unsqueeze(0), kernel.unsqueeze(1), groups=im_hr.shape[0])
    if downsampler.lower() == 'direct':
        im_blur = im_blur[0, :, :, ::sf, ::sf]      # N x c x ...
    elif downsampler.lower() == 'bicubic':
        im_blur = resize(im_blur, scale_factors=1/sf)
    else:
        sys.exit('Please input the corrected downsampler: Direct or Bicubic!')

    return im_blur

def shifted_anisotropic_Gaussian(k_size=21, sf=4, lambda_1=1.2, lambda_2=5., theta=0, shift=True):
    '''
    # modified version of https://github.com/cszn/USRNet/blob/master/utils/utils_sisr.py
    '''
    # set covariance matrix
    Lam = np.diag([lambda_1, lambda_2])
    U = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    sigma = U @ Lam @ U.T                                 # 2 x 2
    inv_sigma = np.linalg.inv(sigma)[None, None, :, :]    # 1 x 1 x 2 x 2

    # set expectation position (shifting kernel for aligned image)
    if shift:
        center = k_size // 2 + 0.5*(sf - k_size % 2)
    else:
        center = k_size // 2

    # Create meshgrid for Gaussian
    X, Y = np.meshgrid(range(k_size), range(k_size))
    Z = np.stack([X, Y], 2).astype(np.float32)[:, :, :, None]                  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - center
    ZZ_t = ZZ.transpose(0,1,3,2)
    ZZZ = -0.5 * np.squeeze(ZZ_t @ inv_sigma  @ ZZ).reshape([1, -1])
    kernel = softmax(ZZZ, axis=1).reshape([k_size, k_size]) # k x k

    # The convariance of the marginal distributions along x and y axis
    s1, s2 = sigma[0, 0], sigma[1, 1]
    # Pearson corrleation coefficient
    rho = sigma[0, 1] / (math.sqrt(s1) * math.sqrt(s2))
    kernel_infos = np.array([s1, s2, rho])   # (3,)

    return kernel, kernel_infos


def imconv_np(im, kernel, padding_mode='reflect', correlate=False):
    '''
    Image convolution or correlation.
    Input:
        im: h x w x c numpy array
        kernel: k x k numpy array
        padding_mode: 'reflect', 'constant' or 'wrap'
    '''
    if kernel.ndim != im.ndim: kernel = kernel[:, :, np.newaxis]

    if correlate:
        out = snd.correlate(im, kernel, mode=padding_mode)
    else:
        out = snd.convolve(im, kernel, mode=padding_mode)

    return out


def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def jpeg_compress(im, qf, chn_in='rgb'):
    '''
    Input:
        im: h x w x 3 array
        qf: compress factor, (0, 100]
        chn_in: 'rgb' or 'bgr'
    Return:
        Compressed Image with channel order: chn_in
    '''
    # transform to BGR channle and uint8 data type
    im_bgr = rgb2bgr(im) if chn_in.lower() == 'rgb' else im
    if im.dtype != np.dtype('uint8'): im_bgr = img_as_ubyte(im_bgr)

    # JPEG compress
    flag, encimg = cv2.imencode('.jpg', im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), qf])
    assert flag
    im_jpg_bgr = cv2.imdecode(encimg, 1)    # uint8, BGR

    # transform back to original channel and the original data type
    im_out = bgr2rgb(im_jpg_bgr) if chn_in.lower() == 'rgb' else im_jpg_bgr
    if im.dtype != np.dtype('uint8'): im_out = img_as_float32(im_out).astype(im.dtype)
    return im_out