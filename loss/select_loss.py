# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:21 PM
# @File    : select_loss.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

from loss.basic_loss.image_loss import *


class Loss:
    def __init__(self, option):
        self.name = option['train']['loss']
        self.loss_char = CharbonnierLoss()
        self.loss_edge = EdgeLoss()

    def __call__(self):
        if self.name.upper() == 'l1'.upper():
            return self.__l1__
        elif self.name.upper() == 'mse'.upper():
            return self.__mse__
        elif self.name.upper() == 'l1_edge'.upper():
            return self.__l1_edge__
        elif self.name.upper() == 'mse_edge'.upper():
            return self.__l2_edge__
        elif self.name.upper() == 'mprnet'.upper():
            return self.__mprnet__
        elif self.name.upper() == 'char'.upper():
            return self.__char__

    def __l1__(self, input, target):
        loss = F.l1_loss(input, target)
        return loss

    def __mse__(self, input, target):
        loss = F.mse_loss(input, target)
        return loss

    def __l1_edge__(self, input, target):
        loss = F.l1_loss(input, target) + self.loss_edge(input, target)
        return loss

    def __l2_edge__(self, input, target):
        loss = F.mse_loss(input, target) + self.loss_edge(input, target)
        return loss

    def __mprnet__(self, input, target):
        loss = self.loss_char(input, target) + self.loss_edge(input, target)
        return loss

    def __char__(self, input, target):
        loss = self.loss_char(input, target)
        return loss
