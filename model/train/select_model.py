# -*- coding: utf-8 -*-
# @Time    : 3/5/23 8:38 PM
# @File    : select_model.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

class Model:
    def __init__(self, option, logger, main_dir):
        self.option = option
        self.logger = logger
        self.main_dir = main_dir
        self.task = option['global_setting']['task']
        self.model = option['global_setting']['model_name']

    def __call__(self):
        if self.task.upper() == 'standard'.upper() or self.model.upper() == 'standard'.upper():
            from model.basic_model.standard_model import Model
            model = Model(self.option, self.logger, self.main_dir)
            return model

        if self.model.upper() == 'vpn'.upper():
            from model.train.vpn_model import Model


        model = Model(self.option, self.logger, self.main_dir)
        return model
