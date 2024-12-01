# -*- coding: utf-8 -*-
# @Time    : 3/5/23 9:01 PM
# @File    : test.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn


import os
import shutil
from tqdm import tqdm
from util.log_util import Logger
# from model.test.test_model import BasicModel
from model.test.vpn_test_model import BasicModel
from util.log_util import Recorder

def test(model, logger):
    test_num = model.dataset_test.__len__()
    psnr, ssim, sam, ergas, q, rmse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # min_rmse = 100.0
    with tqdm(total=test_num) as pbar:
        for i in range(0, test_num):
            _, file = os.path.split(model.dataset_test.input_paths[i])
            data = model.dataset_test.__getitem__(i)
            info = model.test(file, data)
            psnr += info[0]
            ssim += info[1]
            sam += info[2]
            ergas += info[3]
            q += info[4]
            rmse += info[5]
            # if info[5]< min_rmse:
            #     min_rmse = info[5]
            # percep += info[2]
            pbar.set_description('testing ...')
            # pbar.set_postfix(psnr=format(psnr / test_num, '.6f'), ssim=format(ssim / test_num, '.6f'), percep=format(percep / test_num, '.6f'))
            pbar.set_postfix(psnr=format(psnr / test_num, '.6f'), ssim=format(ssim / test_num, '.6f'), sam=format(sam / test_num, '.6f'), ergas=format(ergas / test_num, '.6f'), q=format(q / test_num, '.6f'), rmse=format(rmse / test_num, '.6f'))
            pbar.update(1)
            # logger.info(f'{file}   ----- > psnr = {info[0]: .6f},  ssim = {info[1]: .6f},  perceptual = {info[2]: .6f}')
            logger.info(f'{file}   ----- > PSNR = {info[0]: .6f},  SSIM = {info[1]: .6f}, SAM = {info[2]: .6f}, ERGAS = {info[3]: .6f}, Q = {info[4]: .6f}, RMSE = {info[5]: .6f}')            

    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        '#                                                   Finish Testing                                                          #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    logger.info(
        # f'Metrics(average) : psnr = {psnr / test_num : .6f}, ssim = {ssim / test_num: .6f}, perceptural_loss = {percep / test_num : .6f}')
        f'Metrics(average) : PSNR = {psnr / test_num : .6f}, SSIM = {ssim / test_num: .6f}, SAM = {sam / test_num: .6f}, ERGAS = {ergas / test_num: .6f}, Q = {q / test_num: .6f}')
    # logger.info(f'min_rmse: {min_rmse:.6f}')

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
        '#                                                   Start Testing                                                           #')
    logger.info(
        '# --------------------------------------------------------------------------------------------------------------------------#')
    test(model, logger)
    return True
