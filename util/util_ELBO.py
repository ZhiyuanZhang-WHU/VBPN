import torch
from math import pi, log, sqrt
import torch.distributions as tdist

import util.util_VPN as util_VPN

def cal_kl_gauss_simple(mu_q, mu_p, var_p): return 0.5 * ((mu_q-mu_p)**2/var_p).mean()

def cal_kl_inverse_gamma_simple(beta_q, alpha_p, beta_p):
    out = alpha_p*(beta_p.div(beta_q)-1) + alpha_p*(beta_q.log()-beta_p.log())
    return out.mean()

def cal_likelihood_sisr(x, kernel, sf, mu_q, var_q, alpha_q, beta_q, downsampler):
    zz = mu_q + torch.randn_like(mu_q) * sqrt(var_q)
    zz_blur = util_VPN.conv_multi_kernel_tensor(zz, kernel, sf, downsampler)
    out = 0.5*log(2*pi) +  0.5*(beta_q.log()-alpha_q.digamma()) +  0.5*alpha_q.div(beta_q)*(x-zz_blur)**2
    return out.mean()

def reparameter_inv_gamma(alpha, beta):
    dist_gamma = tdist.gamma.Gamma(alpha, beta)
    out = 1 / dist_gamma.rsample()
    return out

def reparameter_cov_mat(kinfo_est, kappa0, rho_var):
    '''
    Reparameterize kernelo.
    Input:
        kinfo_est: N x 3
    '''
    alpha_k = torch.ones_like(kinfo_est[:, :2]) * (kappa0-1)
    beta_k = kinfo_est[:, :2] * kappa0
    k_var = reparameter_inv_gamma(alpha_k, beta_k)
    k_var1, k_var2 = torch.chunk(k_var, 2, dim=1)    # N x 1, resampled variance along x and y axis
    rho_mean = kinfo_est[:, 2].unsqueeze(1)          # N x 1, mean of the correlation coffecient
    rho = rho_mean + sqrt(rho_var)*torch.randn_like(rho_mean)  # resampled correlation coffecient
    direction = k_var1.detach().sqrt() * k_var2.detach().sqrt() * torch.clamp(rho, min=-1, max=1)   # N x 1
    k_cov = torch.cat([k_var1, direction, direction, k_var2], dim=1).view(-1, 1, 2, 2) # N x 1 x 2 x 2
    return k_cov