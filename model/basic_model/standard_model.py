# -*- coding: utf-8 -*-
# @Time    : 3/5/23 8:34 PM
# @File    : standard_model.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn


import torch
from util.metric_util import standard_psnr_ssim
from model.basic_model.basic_model import BasicModel


class Model(BasicModel):
    def __init__(self, option, logger, main_dir):
        super().__init__(option, logger, main_dir)

    def __feed__(self, data_pair):
        self.optimizer.zero_grad()
        input, target = data_pair
        if self.gpu:
            input, target = [x.cuda() for x in data_pair]
        output = self.net(input)
        self.loss = self.__criterion__(output, target)
        self.loss.backward()
        if self.option['train']['optim']['gradient_max'] > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.option['train']['optim']['gradient_max'])

    def __eval__(self, data_pair):
        with torch.no_grad():
            input, target = data_pair
            if self.gpu:
                input, target = [x.cuda() for x in data_pair]

            output = self.net(input)
        psnr, ssim = standard_psnr_ssim(output, target, mode=self.option['train']['metric_mode'])
        return psnr, ssim