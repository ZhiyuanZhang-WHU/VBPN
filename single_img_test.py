import os
import torch
import cv2 as cv
from skimage import img_as_ubyte
from util.log_util import Recorder
import util.data_util as data_util
from torch.utils.data import DataLoader
from einops.layers.torch import Rearrange
from dataset.realset_single import TestDataSet
from util.metric_util import standard_psnr_ssim, imgvision_psnr_ssim
from loss.basic_loss.image_loss import PerceptralLoss
from util.train_util import resume_state
from thop import profile
from util.data_util import set_random_seed
from net.pansharping.VPN.model_arch import VIRAttResUNetSR as Net
from scipy.io import savemat
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
set_random_seed(20000320)
def __save_tensorAsmat__(save_dir, name, tensor, ms_lr):
        # save_path = os.path.join(self.save_dir, name)

        file_name = os.path.splitext(name)[0]

        matdir = save_dir + '/mat'
        srdir = save_dir + '/ms_sr'
        lr_dir = save_dir + '/ms_lr'
        if not os.path.exists(matdir):
         os.makedirs(matdir)
        if not os.path.exists(srdir):
         os.makedirs(srdir)
        if not os.path.exists(lr_dir):
         os.makedirs(lr_dir)

        bmp_name = file_name + '.bmp'
        
        mat_save_path = os.path.join(save_dir, 'mat', name)


        image = data_util.tensor2image(tensor)
        image = image.squeeze()
        image = img_as_ubyte(image.clip(-1, 1))
        
        rgb_img = image[:, :, 0:3]
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        bmp_save_path = os.path.join(save_dir, 'ms_sr', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img)


        lr = data_util.tensor2image(ms_lr)
        lr = lr.squeeze()
        lr = img_as_ubyte(lr.clip(-1, 1))

        rgb_img = lr[:, :, 0:3]
        rgb_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
        bmp_save_path = os.path.join(save_dir, 'ms_lr', bmp_name)
        cv.imwrite(bmp_save_path, rgb_img)  

        data = {'sr':image, 'lr':lr}

        savemat(mat_save_path, data)

def test(dataset, model):
    save_dir = "Experiment_zw/3"
    with torch.no_grad():
             _, file = os.path.split(dataset.mspath)
             data = dataset.__getitem__()
             ms_lable, pan_label = [x.cuda() for x in data]
             ms_lable = Rearrange('c h w -> (1) c h w')(ms_lable)
             pan_label = Rearrange('c h w -> (1) c h w')(pan_label)
             mu, kinfo_est, sigma_est, alpha_est = model(ms_lable, pan_label, 4)
             __save_tensorAsmat__(save_dir, file, tensor=mu, ms_lr=ms_lable)

        

if __name__ == "__main__":
    set_random_seed(20000320)
    model = Net().cuda().eval()   ########## need to change
    checkpoint = "checkpoint/save_model/model_current_0346.pth"

    model = resume_state(checkpoint, net=model, mode='model')
    val_set = TestDataSet(mspath='',panpath='')


    test(val_set, model)
