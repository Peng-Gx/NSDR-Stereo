import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from utils import *
from torch.utils.data import DataLoader
import gc
# from apex import amp
import cv2
from datasets import sceneflow_listfile

from models.NSDR import NSDR
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt

def disp_to_color(img):
    b,c,h,w = img.shape

    mapMatrix = np.array([[0,0,0,114],[0,0,1,185],[1,0,0,114],[1,0,1,174],
                            [0,1,0,114],[0,1,1,185],[1,1,0,114],[1,1,1,0]])
    bins = mapMatrix[:-1,-1]
    cbins = np.cumsum(bins)
    bins = bins/cbins[-1]
    cbins = cbins[:-1]/cbins[-1]

    cbins_ = np.repeat(np.repeat(np.repeat(cbins.reshape(1,-1,1,1), b, axis=0), h, axis=2), w, axis=3)
    img_ = np.repeat(img, len(cbins), axis=1)
    ind = np.sum(img_>cbins_, axis=1, keepdims=True)
    cbins = np.insert(cbins, 0, 0.)
    bins = 1/bins

    img_left = (img-cbins[ind])*bins[ind]
    img_left = np.repeat(img_left, 3, axis=1)
    img_right = 1-img_left

    color_left = np.concatenate((mapMatrix[ind, 2], mapMatrix[ind, 1], mapMatrix[ind, 0]), axis=1)
    color_right = np.concatenate((mapMatrix[ind+1, 2], mapMatrix[ind+1, 1], mapMatrix[ind+1, 0]), axis=1)
    color_img = color_left*img_right+color_right*img_left
    return color_img


def val():
    avg_test_scalars = AverageMeterDict()
    with tqdm(total=len(TestImgLoader),desc="Test") as pbar:
        for batch_idx, sample in enumerate(TestImgLoader):    
            start_time = time.time()
            loss, scalar_outputs = val_sample(sample)

            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs 
            pbar.update(1)
            pbar.set_postfix({'time':'{:.3f}'.format(time.time() - start_time)})

    avg_test_scalars = avg_test_scalars.mean()
    with open(os.path.join(args.logdir,'result.txt'),'a') as file:
        for k,v in avg_test_scalars.items():
            file.write(f'{k}:{v}\n')
    print("avg_test_scalars", avg_test_scalars)
    gc.collect()

def writeFunctionImg(x=None, y=None, name='sample.png', title="None"):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y,color='g')
    # plt.xlabel('x')
    # plt.ylabel('f(x)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(name)

# test one sample
@make_nograd_func
def val_sample(sample, compute_metrics=True):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    if torch.cuda.is_available() and args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp_gt = disp_gt.cuda()

    h, w = imgL.shape[-2:]
    results = model(imgL, imgR)
    disp_est = results[args.stage]
    if disp_est.shape[-1] != w or disp_est.shape[-2] != h:
        disp_est = F.interpolate(disp_est.unsqueeze(1),size=(h, w),mode='bilinear').squeeze(1)

    if args.dataset=='kitti':
        imgL_name_list = [x.split('/')[-1] for x in sample['left_filename']]
    else:
        imgL_name_list = [x.split('/') for x in sample['left_filename']]
        imgL_name_list = [x[-4]+'_'+x[-3]+'_'+x[-2]+'_'+x[-1] for x in imgL_name_list]

    # #保存分布
    # distribution_path = os.path.join(args.logdir,'distribution')
    # os.makedirs(distribution_path, exist_ok=True)
    # cost = results[-1][-2].squeeze(1)
    # conf = results[-1][-1]
    # for i in range(cost.shape[0]):
    #     x = torch.range(0,cost.shape[1]-1,1).cpu().numpy()
    #     y = cost[i,:,cost.shape[2]//2,cost.shape[3]//2].cpu().numpy()
    #     writeFunctionImg(x,y,os.path.join(distribution_path,imgL_name_list[i]),title="conf:{}".format(conf[i,0,cost.shape[2]//2,cost.shape[3]//2]))

    # #保存置信度
    # confidence_path = os.path.join(args.logdir,'confidence')
    # os.makedirs(confidence_path, exist_ok=True)
    # conf = results[-1][-1]
    # if conf.shape[-1] != w or conf.shape[-2] != h:
    #     conf = F.interpolate(conf,size=(h, w),mode='nearest').squeeze(1)
    # conf = conf.cpu().numpy()*255
    # for i in range(conf.shape[0]):
    #     img = 255-conf[i]
    #     cv2.imwrite(os.path.join(confidence_path, imgL_name_list[i]), img)

    # #保存error
    # error_path = os.path.join(args.logdir,'error')
    # os.makedirs(error_path, exist_ok=True)
    # errormap = disp_error_image_func.apply(disp_est, disp_gt)
    # errormap = errormap.cpu().numpy().transpose(0,2,3,1)*255
    # for i in range(errormap.shape[0]):
    #     img = errormap[i]
    #     cv2.imwrite(os.path.join(error_path, imgL_name_list[i]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # #保存色彩视差图
    # color_disp_path = os.path.join(args.logdir,'color_disp_0')
    # os.makedirs(color_disp_path, exist_ok=True)
    # disp_est_np = disp_est.cpu().numpy()
    # for i in range(disp_est_np.shape[0]):
    #     img = disp_est_np[i]
    #     color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/args.maxdisp
    #     color_img = np.clip(color_img, 0, 1)
    #     color_img = (disp_to_color(color_img)*255).transpose(0,2,3,1)
    #     cv2.imwrite(os.path.join(color_disp_path, 'color_'+imgL_name_list[i]), color_img[0])

    # #保存色彩真值图
    # color_gt_path = os.path.join(args.logdir,'color_gt_0')
    # os.makedirs(color_gt_path, exist_ok=True)
    # disp_gt_np = disp_gt.cpu().numpy()
    # for i in range(disp_gt_np.shape[0]):
    #     img = disp_gt_np[i]
    #     color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/args.maxdisp
    #     color_img = np.clip(color_img, 0, 1)
    #     color_img = (disp_to_color(color_img)*255).transpose(0,2,3,1)
    #     cv2.imwrite(os.path.join(color_gt_path, 'color_'+imgL_name_list[i]), color_img[0])

    #计算保存其他指标
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    if mask.float().mean()<0.01:
        loss=torch.tensor(0, dtype=torch.float32, device=disp_gt.device)
    else:
        loss = F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean')

    scalar_outputs = {"loss": loss}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask)]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask)]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0)]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0)]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0)]

    return tensor2float(loss), tensor2float(scalar_outputs)


def test():
    with tqdm(total=len(TestImgLoader),desc="Test") as pbar:
        for batch_idx, sample in enumerate(TestImgLoader):    
            start_time = time.time()
            test_sample(sample)

            pbar.update(1)
            pbar.set_postfix({'time':'{:.3f}'.format(time.time() - start_time)})
    gc.collect()


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    imgL, imgR= sample['left'], sample['right']
    if torch.cuda.is_available() and args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    h, w = imgL.shape[-2:]
    results = model(imgL, imgR)
    disp_est = results[args.stage]#b,h,w
    if disp_est.shape[-1] != w or disp_est.shape[-2] != h:
        disp_est = F.interpolate(disp_est.unsqueeze(1),size=(h, w),mode='nearest').squeeze(1)
    disp_est_np = disp_est.cpu().numpy()#b,h,w
    
    if args.dataset=='kitti' :
        assert disp_est_np.shape[0] == 1
        disp_est_np = disp_est_np[:,sample['top_pad'][0]:,:]
        disp_est_np = disp_est_np[:,:,0:disp_est_np.shape[2]-sample['right_pad'][0]]

    if args.dataset=='kitti':
        imgL_name_list = [x.split('/')[-1] for x in sample['left_filename']]
    else:
        imgL_name_list = [x.split('/') for x in sample['left_filename']]
        imgL_name_list = [x[-4]+'_'+x[-3]+'_'+x[-2]+'_'+x[-1] for x in imgL_name_list]


    # save confidence map
    confidence_path = os.path.join(args.logdir,'confidence')
    os.makedirs(confidence_path, exist_ok=True)
    conf = results[-1][-1]
    if conf.shape[-1] != w or conf.shape[-2] != h:
        conf = F.interpolate(conf,size=(h, w),mode='nearest').squeeze(1)
    # 转成numpy方便cv操作
    conf = conf.cpu().numpy()*255
    for i in range(conf.shape[0]):
        img = 255-conf[i]
        cv2.imwrite(os.path.join(confidence_path, imgL_name_list[i]), img)

    # save color disparity and gray disparity
    color_disp_path = os.path.join(args.logdir,'color_disp_0')
    os.makedirs(color_disp_path, exist_ok=True)
    disp_path = os.path.join(args.logdir,'disp_0')
    os.makedirs(disp_path, exist_ok=True)
    for i in range(disp_est_np.shape[0]):
        img = disp_est_np[i]
        cv2.imwrite(os.path.join(disp_path, imgL_name_list[i]), np.clip(img*256, 0, 65535).astype(np.uint16))

        #kitti测试集每一张图片有单独的scale，这里使用96或者最大值
        # color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/np.amax(img)
        color_img = img.reshape(1,1,img.shape[-2],img.shape[-1]).astype(np.float64)/96
        color_img = np.clip(color_img, 0, 1)
        color_img = (disp_to_color(color_img)*255).transpose(0,2,3,1)
        cv2.imwrite(os.path.join(color_disp_path, 'color_'+imgL_name_list[i]), color_img[0])

if __name__ == '__main__':
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    parser = argparse.ArgumentParser(description='Neighborhood-Similarity Guided Disparity Refinement in Lightweight Stereo Matching')
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--mode', default='RGB', help='train data color, RGB or L')

    parser.add_argument('--datapath', default="../sub_sceneflow", help='data path')

    parser.add_argument('--kitti15_datapath', default="../SceneFlow/kitti15", help='data path')
    parser.add_argument('--kitti12_datapath', default="../SceneFlow/kitti12", help='data path')
    parser.add_argument('--testlist',default='filenames/kitti12_test.txt', help='testing list')

    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--logdir',default='./testlog', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default='pretrained_model/NSDR_sceneflow.ckpt',help='load the weights from a specific checkpoint')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--cuda', default=True, type=str, help='use cuda to train the model')
    
    parser.add_argument('--stage', type=int, default=0, help='which stage disparity')

    # parse arguments, set seeds
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.logdir, exist_ok=True)

    # create summary logger
    print("creating new summary file")
    current_time=time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    args.logdir=os.path.join(args.logdir,current_time)
    logger = SummaryWriter(args.logdir)

    # save config
    with open(os.path.join(args.logdir,'config.txt'),'w') as file:
        for k,v in vars(args).items():
            file.write(f'{k}:{v}\n')

    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    if args.dataset=='kitti':
        test_dataset = StereoDataset(args.kitti15_datapath, args.kitti12_datapath, args.testlist, False, mode=args.mode)
    else:
        test_dataset = StereoDataset(args.datapath, False, mode=args.mode)
    TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = NSDR()
    model = nn.DataParallel(model)
    if torch.cuda.is_available() and args.cuda:
        model.cuda()

    # load weight
    if args.loadckpt:
        # load the checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt,map_location=torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda'))
        model_dict = model.state_dict()
        pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
        model_dict.update(pre_dict) 
        model.load_state_dict(model_dict)
        
    if args.dataset == 'kitti':
        test()
    else:
        val()