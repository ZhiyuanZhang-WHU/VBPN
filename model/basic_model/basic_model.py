# -*- coding: utf-8 -*-
# @Time    : 3/5/23 7:40 PM
# @File    : basic_model.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import os
import torch
from net.select_net import Net
from loss.select_loss import Loss
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataset.select_dataset import DataSet
from util.optim_util import Optimizer, Scheduler
from warmup_scheduler import GradualWarmupScheduler
from util.train_util import Early_Stop, resume_state, WeightInit


class BasicModel:
    def __init__(self, option, logger, main_dir):
        batch_size = option['train']['batch_size']
        num_workers = option['train']['num_worker']
        total_iteration = int(option['train']['iteration'])
        model_save_dir = os.path.join(main_dir, option['directory']['save_model'])
        runlog_save_dir = os.path.join(main_dir, option['directory']['runlog'])
        
        self.writer = SummaryWriter(runlog_save_dir)
        self.option = option
        self.logger = logger
        self.batch_size = batch_size
        
        self.patience = option['train']['patience']
        self.gpu = option['global_setting']['gpu']
        """the dataloader of train and test"""
        self.dataset_train, self.dataset_test = DataSet(option)()
        self.loader_train = DataLoader(self.dataset_train, batch_size=batch_size,
                                       num_workers=num_workers, shuffle=True)
        self.loader_test = DataLoader(self.dataset_test, batch_size=1, num_workers=0, shuffle=False)
        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')
        self.logger.info(
            '#                    The DataLoader for train and validation has been loaded to the memory                                  #')
        self.logger.info(
            '# --------------------------------------------------------------------------------------------------------------------------#')
        """ early stop strategy"""
        self.early_stopper = Early_Stop(logger, patience=self.patience, verbose=True, delta=0, save_dir=model_save_dir)

        """network, optimizer, scheduler"""

        self.net = Net(option)()
        if self.option['global_setting']['resume'] == False and option['train']['init']['state']:
            weight_init_function = WeightInit(name=option['train']['init']['name'])()
            self.net.apply(weight_init_function)

        self.optimizer = None
        self.scheduler = None
        self.epoch_begin = 1
        self.epoch_end = total_iteration // (self.dataset_train.__len__() // batch_size)

        self.__optimizer__()
        self.__scheduler__()
        self.__resume__()
        if self.gpu:
            self.net = self.net.cuda()

        if len(self.option['train']['loss']) != 0:
            self.__criterion__ = Loss(self.option)()

    def __resume__(self):
        resume = self.option['global_setting']['resume']
        checkpoint = resume['checkpoint']
        if resume['state']:
            if resume['mode'].upper() == 'all'.upper():
                self.epoch_begin, self.net, self.optimizer, self.scheduler = resume_state(
                    checkpoint, self.net, self.optimizer, self.scheduler, mode=resume['mode']
                )
            else:
                self.net = resume_state(
                    checkpoint, self.net, self.optimizer, self.scheduler, mode=resume['mode']
                )

    def __optimizer__(self):
        self.optimizer = Optimizer(model=self.net, option=self.option)()

    def __scheduler__(self):
        if self.option['train']['scheduler']['state']:
            self.scheduler = Scheduler(optimizer=self.optimizer, option=self.option)()
            warmup = self.option['train']['scheduler']['warmup']
            if warmup['state']:
                self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=warmup['multiplier'],
                                                        total_epoch=warmup['warmup_epoch'],
                                                        after_scheduler=self.scheduler)

    def __step_optimizer__(self):
        self.optimizer.step()

    def __step_scheduler__(self):
        if self.option['train']['scheduler']['state']:
            self.scheduler.step()
