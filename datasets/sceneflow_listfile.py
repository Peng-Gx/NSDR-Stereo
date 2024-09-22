import torch.utils.data as data

from PIL import Image
import os
import os.path
import glob
import numpy as np

import pdb


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#传入sceneflow的路径，sceneflow目录下分别是monkaa，driving和flyingthings
def dataloader(filepath,part='all'):

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    test_left_img = []
    test_right_img = []
    test_left_disp = []

    # # MONKAAS ##
    monkaa_path = os.path.join(filepath, 'monkaa', 'frames_finalpass')
    monkaa_disp = os.path.join(filepath, 'monkaa', 'disparity')
    monkaa_left_img = []
    monkaa_right_img = []
    monkaa_left_disp = []

    if os.path.exists(monkaa_path) and os.path.exists(monkaa_disp):
        monkaa_dir = os.listdir(monkaa_path)
        for dd in monkaa_dir:
            for im in os.listdir(os.path.join(monkaa_path, dd, 'left')):
                if is_image_file(os.path.join(monkaa_path, dd, 'left', im)):
                    monkaa_left_img.append(os.path.join(monkaa_path, dd, 'left', im))
                    monkaa_left_disp.append(os.path.join(monkaa_disp, dd, 'left', im.split(".")[0]+'.pfm'))

            for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
                if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                    monkaa_right_img.append(monkaa_path+'/'+dd+'/right/'+im)
        monkaa_left_img.sort()
        monkaa_right_img.sort()
        monkaa_left_disp.sort()
    
    print(f"len of monkaa_img_pair is {len(monkaa_left_img)}")


    # # FLYINGTHINGS 3D ##
    flying_path = os.path.join(filepath, 'flyingthings3d_final', 'frames_finalpass', 'TRAIN')
    flying_disp = os.path.join(filepath, 'flyingthings3d_final', 'disparity', 'TRAIN')
    flying_left_img, flying_right_img, flying_left_disp = [], [], []
    flying_left_img_val, flying_right_img_val, flying_left_disp_val = [], [], []

    if os.path.exists(flying_path) and os.path.exists(flying_disp):
        flying_dir = os.listdir(flying_path)
        for dd in flying_dir:
            for nn in os.listdir(os.path.join(flying_path, dd)):
                for im in os.listdir(os.path.join(flying_path, dd, nn, 'left')):
                    if is_image_file(os.path.join(flying_path, dd, nn, 'left', im)):
                        flying_left_img.append(os.path.join(flying_path, dd, nn, 'left', im))
                        flying_left_disp.append(os.path.join(flying_disp, dd, nn, 'left', im.split(".")[0]+'.pfm'))
                for im in os.listdir(os.path.join(flying_path, dd, nn, 'right')):
                    if is_image_file(os.path.join(flying_path, dd, nn, 'right', im)):
                        flying_right_img.append(os.path.join(flying_path, dd, nn, 'right', im))

        flying_path_val = os.path.join(filepath, 'flyingthings3d_final', 'frames_finalpass', 'TEST')
        flying_disp_val = os.path.join(filepath, 'flyingthings3d_final', 'disparity', 'TEST')

        flying_dir_val = os.listdir(flying_path_val)
        for dd in flying_dir_val:
            for nn in os.listdir(os.path.join(flying_path_val, dd)):
                for im in os.listdir(os.path.join(flying_path_val, dd, nn, 'left')):
                    if is_image_file(os.path.join(flying_path_val, dd, nn, 'left', im)):
                        flying_left_img_val.append(os.path.join(flying_path_val, dd, nn, 'left', im))
                        flying_left_disp_val.append(os.path.join(flying_disp_val, dd, nn, 'left', im.split(".")[0]+'.pfm'))
                for im in os.listdir(os.path.join(flying_path_val, dd, nn, 'right')):
                    if is_image_file(os.path.join(flying_path_val, dd, nn, 'right', im)):
                        flying_right_img_val.append(os.path.join(flying_path_val, dd, nn, 'right', im))

        flying_left_img.sort()
        flying_right_img.sort()
        flying_left_disp.sort()
        flying_left_img_val.sort()
        flying_right_img_val.sort()
        flying_left_disp_val.sort()
    
    print(f"len of flying_img_pair is {len(flying_left_img)}")
    print(f"len of flying_img_pair_val is {len(flying_left_img_val)}")


    # # DRIVING ##
    driving_dir = os.path.join(filepath, 'driving', 'frames_finalpass')
    driving_disp = os.path.join(filepath, 'driving', 'disparity')
    driving_left_img = []
    driving_right_img = []
    driving_left_disp = []
    subdir1 = ['35mm_focallength', '15mm_focallength']
    subdir2 = ['scene_backwards', 'scene_forwards']
    subdir3 = ['fast', 'slow']

    if os.path.exists(driving_dir) and os.path.exists(driving_disp):
        for i in subdir1:
            for j in subdir2:
                for k in subdir3:
                    imm_l = os.listdir(os.path.join(driving_dir, i, j, k, 'left'))
                    for im in imm_l:
                        if is_image_file(os.path.join(driving_dir, i, j, k, 'left', im)):
                            driving_left_img.append(os.path.join(driving_dir, i, j, k, 'left', im))
                            driving_left_disp.append(os.path.join(driving_disp, i, j, k, 'left', im.split(".")[0]+'.pfm'))

                        if is_image_file(os.path.join(driving_dir, i, j, k, 'right', im)):
                            driving_right_img.append(os.path.join(driving_dir, i, j, k, 'right', im))
        driving_left_img.sort()
        driving_right_img.sort()
        driving_left_disp.sort()

    print(f"len of driving_img_pair is {len(driving_left_img)}")

    #要保证flying目录结构完整
    if part=='all':
        all_left_img=monkaa_left_img+flying_left_img+driving_left_img
        all_right_img=monkaa_right_img+flying_right_img+driving_right_img
        all_left_disp=monkaa_left_disp+flying_left_disp+driving_left_disp
        test_left_img =flying_left_img_val
        test_right_img =flying_right_img_val
        test_left_disp =flying_left_disp_val
    #要保证driving目录结构完整
    elif part=='driving':
        all_left_img=driving_left_img[:int(len(driving_left_img)*0.95)]
        all_right_img=driving_right_img[:int(len(driving_right_img)*0.95)]
        all_left_disp=driving_left_disp[:int(len(driving_left_disp)*0.95)]
        test_left_img=driving_left_img[int(len(driving_left_img)*0.95):]
        test_right_img=driving_right_img[int(len(driving_right_img)*0.95):]
        test_left_disp=driving_left_disp[int(len(driving_left_disp)*0.95):]

    return all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp


if __name__=='__main__':
    files=dataloader('../sub_sceneflow')
    print(len(files[0]))