
import os
import torch
import cv2 as cv
from net.select_net import Net
from skimage import img_as_ubyte
from util.log_util import Recorder
import util.data_util as data_util
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from dataset.select_dataset import DataSet
from util.metric_util import standard_psnr_ssim, imgvision_psnr_ssim
from loss.basic_loss.image_loss import PerceptralLoss
from util.train_util import resume_state
from thop import profile

from scipy.io import savemat


class BasicModel:
    def __init__(self, option, logger, main_dir):
        self.option = option
        self.logger = logger
        self.main_dir = main_dir
        self.recoder = Recorder(option)
        self.save = option['test']['save']
        self.gpu = option['test']['gpu']
        # self.mode = option['network']['mode']
        self.mode = option['test']['metric_mode']
        self.save_dir = os.path.join(main_dir, option['directory']['vision'])
        self.net = Net(option=option)()
        _, self.dataset_test = DataSet(option)()
        self.loader_test = DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=0)
        # self.percepLoss = PerceptralLoss()
        


        self.__resume__()
        if self.gpu:
            self.net = self.net.cuda()
            # self.percepLoss = self.percepLoss.cuda()

        logger.info("Every Thing has been prepared . ")

    def __resume__(self):
        mode = self.option['global_setting']['resume']['mode']
        checkpoint = self.option['global_setting']['resume']['checkpoint']
        self.net = resume_state(checkpoint, net=self.net, mode=mode)

    def __save_tensor__(self, name, tensor):
        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(-1, 1))
        save_path = os.path.join(self.save_dir, name)
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imwrite(save_path, image)

    # save mat
    def __save_tensorAsmat__(self, name, tensor, gt, ms_lr, pan_n):
        # save_path = os.path.join(self.save_dir, name)

        file_name = os.path.splitext(name)[0]
        bmp_name = file_name + '.bmp'
        
        mat_save_path = os.path.join(self.save_dir, 'mat', name)


        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(-1, 1))
        
        rgb_img = image[:, :, 0:3]
        
        bmp_save_path = os.path.join(self.save_dir, 'ms_sr', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img)


        lr = data_util.tensor2image(ms_lr)
        lr = lr.squeeze()
        lr = img_as_ubyte(lr.clip(-1, 1))

        rgb_img = lr[:, :, 0:3]
        
        bmp_save_path = os.path.join(self.save_dir, 'ms_lr', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img)  


        gt = data_util.tensor2image(gt)
        gt = gt.squeeze()
        gt = img_as_ubyte(gt.clip(-1, 1))

        rgb_img = gt[:, :, 0:3]
        
        bmp_save_path = os.path.join(self.save_dir, 'ms_label', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img) 

        pan_n = data_util.tensor2image(pan_n)
        pan_n = pan_n.squeeze()
        pan_n = img_as_ubyte(pan_n.clip(-1, 1))

        #rgb_img = pan_n[:, :, 0:3]
        bmp_save_path = os.path.join(self.save_dir, 'pan_n', bmp_name)
        cv.imwrite(bmp_save_path, pan_n)        

        data = {'gt':gt , 'sr':image, 'lr':lr}

        savemat(mat_save_path, data)

    def test(self, name, data_pair):
        self.net.eval()
        with torch.no_grad():
            import time
            
  
            
            if self.gpu:
                ms_lable, ms_lr, kinfo_gt, \
                    pan_label, pan_n = [x.cuda() for x in data_pair]
            else:
                ms_lable, ms_lr, kinfo_gt, \
                    pan_label, pan_n = data_pair

            ms_lable = Rearrange('c h w -> (1) c h w')(ms_lable)
            ms_lr = Rearrange('c h w -> (1) c h w')(ms_lr)
            pan_n = Rearrange('c h w -> (1) c h w')(pan_n)
            time_start = time.time()
            mu, kinfo_est, sigma_est, alpha_est = self.net(ms_lr, pan_n, self.option['dataset']['sf']) 
            time_end = time.time()
            times = time_end-time_start
            # times += times
            # print(f'-----------excute expend {time_end-time_start}-------------\n')
            # flops, params = profile(self.net, (ms_lr,pan_n,4))
            # print('flops: ', flops, 'params: ', params) 
            # f = open('Experiment/test/pansharping/flops/test.txt','a')
            # f.write(str(times)+ '\n')
            # f.close()              

            # input, target = [x for x in data_pair]
            # input = Rearrange('c h w -> (1) c h w')(input)
            # target = Rearrange('c h w -> (1) c h w')(target)
            # if self.gpu:
            #     input, target = input.cuda(), target.cuda()
            # output = self.net(input)

        if self.save:
            # self.__save_tensor__(name, tensor=output)
            self.__save_tensorAsmat__(name, tensor=mu, gt=ms_lable, ms_lr=ms_lr, pan_n= pan_n)

        # psnr, ssim = standard_psnr_ssim(input=mu, target=ms_lable, mode=self.mode)
        PSNR, SSIM, SAM, ERGAS, Q, RMSE = imgvision_psnr_ssim(input = mu, target = ms_lable)
        # with torch.no_grad():
        #     if output.size(1) == 1:
        #         output = torch.cat([output, output, output], dim=1)
        #         target = torch.cat([target, target, target], dim=1)
        #     percep_loss = self.percepLoss(output, target)
        # return psnr, ssim, percep_loss.item()
        return PSNR, SSIM, SAM, ERGAS, Q, RMSE
