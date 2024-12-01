# -*- coding: utf-8 -*-
# @Time    : 3/5/23 7:42 PM
# @File    : select_net.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

class Net:
    def __init__(self, option):
        self.task = option['network']['task']
        self.info_net = option['network']

    def __call__(self):
        if self.task.upper() == 'pansharping'.upper():
            from net.pansharping.select import select_network 

        network = select_network(self.info_net)
        return network