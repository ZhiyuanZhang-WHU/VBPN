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
from util.metric_util import standard_psnr_ssim, imgvision_psnr_ssim, D_lamda, D_s
from loss.basic_loss.image_loss import PerceptralLoss
from util.train_util import resume_state

from scipy.io import savemat
from osgeo import gdal


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
        self.loader_test = DataLoader(self.dataset_test, batch_size=32, shuffle=False, num_workers=0)
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
    def __save_tensorAsmat__(self, name, tensor, ms_lr):
        # save_path = os.path.join(self.save_dir, name)

        file_name = os.path.splitext(name)[0]
        bmp_name = file_name + '.bmp'
        
        mat_save_path = os.path.join(self.save_dir, 'mat', name)


        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(-1, 1))
        
        rgb_img = image[:, :, 0:3]
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        bmp_save_path = os.path.join(self.save_dir, 'ms_sr', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img)


        lr = data_util.tensor2image(ms_lr)
        lr = lr.squeeze()
        lr = img_as_ubyte(lr.clip(-1, 1))

        rgb_img = lr[:, :, 0:3]
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        bmp_save_path = os.path.join(self.save_dir, 'ms_lr', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img)  


        data = {'sr':image, 'lr':lr}

        savemat(mat_save_path, data)

    # save tiff
    def __save_tensorAstiff__(self, name, tensor, ms_lr):
        # save_path = os.path.join(self.save_dir, name)
        def savetiff(path,img):
            pixelWidth = 1.0
            pixelHeight = -1.0

            cols = img.shape[1]
            rows = img.shape[0]
            if len(img.shape) == 3:
                bands = img.shape[2]
            else:
                bands = 1
            originX = 0
            originY = 0
            driver = gdal.GetDriverByName('GTiff')

            outRaster = driver.Create(path, cols, rows, bands, gdal.GDT_UInt16)
            outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
            #开始写入
            if bands==1:
                outband = outRaster.GetRasterBand(1)
                outband.WriteArray(img[:, :,0])
            else:
                for i in range(bands):
                    outRaster.GetRasterBand(i + 1).WriteArray(img[:,:,i])

        file_name = os.path.splitext(name)[0]
        tiff_name = file_name + '.tiff'
        
        # mat_save_path = os.path.join(self.save_dir, 'mat', name)


        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(-1, 1))
        

        tiff_save_path = os.path.join(self.save_dir, 'ms_sr', tiff_name)
        savetiff(tiff_save_path, image)


        lr = data_util.tensor2image(ms_lr)
        lr = lr.squeeze()
        lr = img_as_ubyte(lr.clip(-1, 1))


        tiff_save_path = os.path.join(self.save_dir, 'ms_lr', tiff_name)
        savetiff(tiff_save_path, lr)
    

    def test(self, name, data_pair):
        self.net.eval()
        with torch.no_grad():

            if self.gpu:
                ms_lable, ms_lr, kinfo_gt, \
                    pan_label, pan_n = [x.cuda() for x in data_pair]
            else:
                ms_lable, ms_lr, kinfo_gt, \
                    pan_label, pan_n = data_pair

            ms_lable = Rearrange('c h w -> (1) c h w')(ms_lable)
            ms_lr = Rearrange('c h w -> (1) c h w')(ms_lr)
            pan_n = Rearrange('c h w -> (1) c h w')(pan_n)
            pan_label = Rearrange('c h w -> (1) c h w')(pan_label)

            mu, kinfo_est, sigma_est, alpha_est = self.net(ms_lable, pan_label, self.option['dataset']['sf'])                

            # input, target = [x for x in data_pair]
            # input = Rearrange('c h w -> (1) c h w')(input)
            # target = Rearrange('c h w -> (1) c h w')(target)
            # if self.gpu:
            #     input, target = input.cuda(), target.cuda()
            # output = self.net(input)

        if self.save:
            # self.__save_tensor__(name, tensor=output)
            self.__save_tensorAsmat__(name, tensor=mu, ms_lr=ms_lable)
            # self.__save_tensorAstiff__(name, tensor=mu, ms_lr=ms_lable)

        D_lambda = D_lamda(mu, ms_lable)
        Ds = D_s(mu, ms_lable, pan_label)
        QNR = (1-D_lambda)*(1-Ds)

        return D_lambda, Ds, QNR
