import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''
    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None  # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.HR_env, self.paths_HR = util.get_image_paths(opt['data_type'], opt['dataroot_HR'])
            self.LR_env, self.paths_LR = util.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR path is empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = [1]

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = util.read_img(self.HR_env, HR_path)

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = util.read_img(self.LR_env, LR_path)
        else:  # down-sampling on-the-fly
            if self.opt['phase'] == 'train':
                # force to 3 channels
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            img_LR = cv2.resize(img_HR, dsize=(int(H / scale), int(W / scale)), interpolation=cv2.INTER_CUBIC)
            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)

        if self.opt['phase'] == 'train':
            # augmentation - flip, rotate
            img_LR, img_HR = util.augment([img_LR, img_HR], self.opt['use_flip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def __len__(self):
        return len(self.paths_HR)
