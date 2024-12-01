# -*- coding: utf-8 -*-
# @Time    : 3/5/23 8:38 PM
# @File    : train.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import os
import shutil
from tqdm import tqdm
from util.log_util import Recorder, Logger
from model.train.select_model import Model


def train(model, option, logger):
    best_psnr = 0.0
    val_num = model.dataset_test.__len__()
    for epoch in range(model.epoch_begin, model.epoch_end + 1):
        loop_train = tqdm(model.loader_train, desc='training')
        for _, data in enumerate(loop_train, 0):
            model.net.train()
            model.__feed__(data)
            model.__step_optimizer__()
            loop_train.set_description(f"Epoch [{epoch} / {model.epoch_end}], lr = {model.optimizer.state_dict()['param_groups'][0]['lr'] : .6f}")
            loop_train.set_postfix(loss=model.loss.item())

        if epoch % option["train"]["freq_valid"] == 0:
            model.net.eval()
            psnr, ssim = 0.0, 0.0
            loop_valid = tqdm(model.loader_test, desc="valid")
            for _, data in enumerate(loop_valid, 0):
                info = model.__eval__(data)
                psnr += info[0]
                ssim += info[1]
                loop_valid.set_description(
                    f"Epoch [{epoch} / {model.epoch_end}], lr = {model.optimizer.state_dict()['param_groups'][0]['lr']: .6f}")
                loop_valid.set_postfix(BestPSNR=format(best_psnr, '.4f'), CurrentPSNR=format(psnr / val_num, '.4f'))
            
            best_psnr = max(best_psnr, psnr / val_num)
            model.writer.add_scalar("validation / psnr", psnr / val_num, epoch)
            model.early_stopper.stop_metric(epoch, model.net, model.optimizer, psnr / val_num)

            if model.early_stopper.early_stop:
                logger.info("early stop")
                break

        model.__step_scheduler__()

    logger.info('# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info('#                                                   Finish Trainin                                                          #')
    logger.info('# --------------------------------------------------------------------------------------------------------------------------#')

def inlet(option, args):
    recorder = Recorder(option=option)
    recorder.__call__()
    _, yamlfile = os.path.split(args.yaml)
    shutil.copy(args.yaml, os.path.join(recorder.main_record, yamlfile))
    logger = Logger(log_dir=recorder.main_record)()
    logger.info('# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info('#                                                   Start Training                                                          #')
    logger.info('# --------------------------------------------------------------------------------------------------------------------------#')
    model = Model(option, logger, main_dir=recorder.main_record)()
    train(model, option, logger)