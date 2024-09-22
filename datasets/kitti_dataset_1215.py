import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from datasets.data_io import get_transform, read_all_lines, pfm_imread
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt


class KITTIDataset(Dataset):
    def __init__(self, kitti15_datapath, kitti12_datapath, list_filename, training, crop_w=576, crop_h=288, mode='RGB'):
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.datapath_15 = kitti15_datapath
        self.datapath_12 = kitti12_datapath
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        self.mode = mode
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        left_name = self.left_filenames[index].split('/')[1]
        if left_name.startswith('image'):
            self.datapath = self.datapath_15
        else:
            self.datapath = self.datapath_12

        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            # crop_w, crop_h = 512, 256
            crop_w, crop_h = self.crop_w, self.crop_h
            # crop_w, crop_h = 960, 320
            # crop_w, crop_h = 736, 320

            x1 = random.randint(0, w - crop_w)
            # y1 = random.randint(0, h - crop_h)
            if  random.randint(0, 10) >= int(8):
                y1 = random.randint(0, h - crop_h)
            else:
                y1 = random.randint(int(0.2 * h), h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform(training=self.training)
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

            # normalize
            processed = get_transform(training=self.training)
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

            left_img = left_img.numpy()
            right_img = right_img.numpy()

            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)


            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}

