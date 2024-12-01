# -*- coding: utf-8 -*-
# @Time    : 3/5/23 6:02 PM
# @File    : select_dataset.py
# @Author  : lianghao
# @Email   : lianghao@whu.edu.cn


import numpy as np

def select_dataset(info_dataset, patch_size, mode):
    clip = info_dataset['clip']
    name = info_dataset['name']
    task = info_dataset['task']['noise']
    train_dataset, test_dataset = (None, None)

    if name.upper() == 'vpn'.upper():
        from dataset.pansharping.vpn import TrainDataSet, TestDataSet
        # need add: noise level
        if 'train' in info_dataset.keys():
            # train_noise_levels = get_noise_level(info_dataset, action='train')

            train_dataset = TrainDataSet(input_dir=info_dataset['train']['target'], 
                                         ms_patch_size=info_dataset['train']['ms_patch_size'],
                                         sf=info_dataset['sf'], 
                                         ms_k_size=info_dataset['train']['ms_k_size'],
                                         ms_kernel_shift=info_dataset['train']['ms_kernel_shift'], 
                                         downsampler=info_dataset['downsampler'],
                                         ms_noise_level=info_dataset['train']['ms_noise_level'],
                                         ms_noise_jpeg=info_dataset['train']['ms_noise_jpeg'],
                                         ms_add_jpeg=info_dataset['train']['ms_add_jpeg'],
                                         pan_clip=info_dataset['train']['pan_clip'],
                                         pan_nlevels=info_dataset['train']['pan_nlevels'],
                                         pan_task=info_dataset['train']['pan_task'])

            # train_dataset = TrainDataSet(input_dir=info_dataset['train']['target'], patch_size=patch_size,
            #                              levels=train_noise_levels, task=task,
            #                              mode=mode, clip=clip)

        # test_noise_levels = get_noise_level(info_dataset, action='test')
        test_dataset = TestDataSet(input_dir=info_dataset['test']['target'], 
                                   ms_patch_size=info_dataset['test']['ms_patch_size'],
                                   sf=info_dataset['sf'], 
                                   ms_k_size=info_dataset['test']['ms_k_size'],
                                   ms_kernel_shift=info_dataset['test']['ms_kernel_shift'], 
                                   downsampler=info_dataset['downsampler'],
                                   ms_noise_type=info_dataset['test']['ms_noise_type'],
                                   pan_clip=info_dataset['test']['pan_clip'],
                                   pan_nlevels=info_dataset['test']['pan_nlevels'],
                                   pan_task=info_dataset['test']['pan_task'])


    if name.upper() == 'pn'.upper():
        from dataset.pansharping.vpn import TrainDataSet, TestDataSet
        # need add: noise level
        if 'train' in info_dataset.keys():
            # train_noise_levels = get_noise_level(info_dataset, action='train')

            train_dataset = TrainDataSet(input_dir=info_dataset['train']['target'], 
                                         ms_patch_size=info_dataset['train']['ms_patch_size'],
                                         sf=info_dataset['sf'], 
                                         ms_k_size=info_dataset['train']['ms_k_size'],
                                         ms_kernel_shift=info_dataset['train']['ms_kernel_shift'], 
                                         downsampler=info_dataset['downsampler'],
                                         ms_noise_level=info_dataset['train']['ms_noise_level'],
                                         ms_noise_jpeg=info_dataset['train']['ms_noise_jpeg'],
                                         ms_add_jpeg=info_dataset['train']['ms_add_jpeg'],
                                         pan_clip=info_dataset['train']['pan_clip'],
                                         pan_nlevels=info_dataset['train']['pan_nlevels'],
                                         pan_task=info_dataset['train']['pan_task'])

            # train_dataset = TrainDataSet(input_dir=info_dataset['train']['target'], patch_size=patch_size,
            #                              levels=train_noise_levels, task=task,
            #                              mode=mode, clip=clip)

        # test_noise_levels = get_noise_level(info_dataset, action='test')
        test_dataset = TestDataSet(input_dir=info_dataset['test']['target'], 
                                   ms_patch_size=info_dataset['test']['ms_patch_size'],
                                   sf=info_dataset['sf'], 
                                   ms_k_size=info_dataset['test']['ms_k_size'],
                                   ms_kernel_shift=info_dataset['test']['ms_kernel_shift'], 
                                   downsampler=info_dataset['downsampler'],
                                   ms_noise_type=info_dataset['test']['ms_noise_type'],
                                   pan_clip=info_dataset['test']['pan_clip'],
                                   pan_nlevels=info_dataset['test']['pan_nlevels'],
                                   pan_task=info_dataset['test']['pan_task'])

    return train_dataset, test_dataset


# def get_noise_level(info_dataset, action='train'):
#     if isinstance(info_dataset[action]['levels'], list):
#         noise_levels = info_dataset[action]['levels']
#     else:
#         noise_levels = np.arange(info_dataset[action]['levels']['begin'],
#                                  info_dataset[action]['levels']['end'] + info_dataset[action]['levels']['step'],
#                                  info_dataset[action]['levels']['step'])
#     return noise_levels
