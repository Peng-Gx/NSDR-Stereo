import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, read_all_lines, pfm_imread
from .sceneflow_listfile import dataloader
import torch

class SceneFlowDatset(Dataset):
    # def __init__(self, datapath, list_filename, training):
    #     self.datapath = datapath
    #     self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
    #     self.training = training

    # def load_path(self, list_filename):
    #     lines = read_all_lines(list_filename)
    #     splits = [line.split() for line in lines]
    #     left_images = [x[0] for x in splits]
    #     right_images = [x[1] for x in splits]
    #     disp_images = [x[2] for x in splits]
    #     return left_images, right_images, disp_images

    def __init__(self,scenflow_datapath,training=False,part='all',mode='RGB'):
        filelists=dataloader(scenflow_datapath,part)
        self.training=training
        self.mode = mode
        if self.training:
            self.left_filenames=filelists[0]
            self.right_filenames=filelists[1]
            self.disp_filenames=filelists[2]
        else:
            self.left_filenames=filelists[3]
            self.right_filenames=filelists[4]
            self.disp_filenames=filelists[5]
        

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(self.left_filenames[index])
        right_img = self.load_image(self.right_filenames[index])
        disparity = self.load_disp(self.disp_filenames[index])

        if self.training:
            w, h = left_img.size
            # crop_w, crop_h = 512, 256
            crop_w, crop_h = 576, 288

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform(mode=self.mode)
            left_img = processed(left_img)
            right_img = processed(right_img)

            if self.mode == 'L':
                left_img = torch.mean(left_img, dim=0, keepdim=True)
                right_img = torch.mean(right_img, dim=0, keepdim=True)
                zero_img = torch.zeros_like(left_img)
                # zero_img = left_img
                left_img = torch.concat([left_img,zero_img, zero_img], dim=0)
                # zero_img = right_img
                right_img = torch.concat([right_img,zero_img, zero_img], dim=0)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size
            crop_w, crop_h = 960, 512

            left_img = left_img.crop((w - crop_w, h - crop_h, w, h))
            right_img = right_img.crop((w - crop_w, h - crop_h, w, h))
            disparity = disparity[h - crop_h:h, w - crop_w: w]

            # to tensor, normalize
            processed = get_transform(mode=self.mode)
            left_img = processed(left_img)
            right_img = processed(right_img)

            if self.mode == 'L':
                left_img = torch.mean(left_img, dim=0, keepdim=True)
                right_img = torch.mean(right_img, dim=0, keepdim=True)
                zero_img = torch.zeros_like(left_img)
                # zero_img = left_img
                left_img = torch.concat([left_img,zero_img, zero_img], dim=0)
                # zero_img = right_img
                right_img = torch.concat([right_img,zero_img, zero_img], dim=0)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename": self.left_filenames[index]}
