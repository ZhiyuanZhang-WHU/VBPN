# -*- coding: utf-8 -*-
# @Time    : 3/5/23 9:01 PM
# @File    : real.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn

import os
import shutil
from tqdm import tqdm
from util.log_util import Logger
from model.test.vpn_real_model import BasicModel
from util.log_util import Recorder


def realtest(model, logger):
    test_num = model.dataset_test.__len__()
    d_lambda, d_s, qnr = 0.0, 0.0, 0.0
    with tqdm(total=test_num) as pbar:
        pbar.set_description('running ...')
        for i in range(0, test_num):
            _, file = os.path.split(model.dataset_test.input_paths[i])
            data = model.dataset_test.__getitem__(i)
            info = model.test(file, data)
            d_lambda += info[0]
            d_s += info[1]
            qnr += info[2]
            pbar.set_description('testing ...')
            # pbar.set_postfix(psnr=format(psnr / test_num, '.6f'), ssim=format(ssim / test_num, '.6f'), percep=format(percep / test_num, '.6f'))
            pbar.set_postfix(d_lambda=format(d_lambda / test_num, '.6f'), d_s=format(d_s / test_num, '.6f'), qnr=format(qnr / test_num, '.6f'))

            pbar.update(1)
            logger.info(f'{file}   ----- > D_lamda = {info[0]: .6f},  D_s = {info[1]: .6f}, QNR = {info[2]: .6f}')  

    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Finish Testing                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        # f'Metrics(average) : psnr = {psnr / test_num : .6f}, ssim = {ssim / test_num: .6f}, perceptural_loss = {percep / test_num : .6f}')
        f'Metrics(average) : D_lamda = {d_lambda / test_num : .6f}, D_s = {d_s / test_num: .6f}, QNR = {qnr / test_num: .6f}')


def inlet(option, args):
    recorder = Recorder(option=option)
    recorder()
    _, yamlfile = os.path.split(args.yaml)
    shutil.copy(args.yaml, os.path.join(recorder.main_record, yamlfile))
    logger = Logger(log_dir=recorder.main_record)()
    model = BasicModel(option, logger, main_dir=recorder.main_record)
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   tart Testing                                                            #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    realtest(model, logger)
    return True
