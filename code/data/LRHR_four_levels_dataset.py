import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util


class LRHRFourLevelsDataset(data.Dataset):
    '''
    Read LR, HR and intermediate target image groups.
    If only HR image is provided, generate LR image on-the-fly.
    The group is matched by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LRHRFourLevelsDataset, self).__init__()
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
            self.D1_env, self.paths_D1 = util.get_image_paths(opt['data_type'], opt['dataroot_D1'])
            self.D2_env, self.paths_D2 = util.get_image_paths(opt['data_type'], opt['dataroot_D2'])
            self.D3_env, self.paths_D3 = util.get_image_paths(opt['data_type'], opt['dataroot_D3'])

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
            D1_path = self.paths_D1[index]
            D2_path = self.paths_D2[index]
            D3_path = self.paths_D3[index]
            img_LR = util.read_img(self.LR_env, LR_path)
            img_D1 = util.read_img(self.D1_env, D1_path)
            img_D2 = util.read_img(self.D2_env, D2_path)
            img_D3 = util.read_img(self.D3_env, D3_path)
        else:  # down-sampling on-the-fly
            if self.opt['phase'] == 'train':
                # force to 3 channels
                if img_HR.ndim == 2:
                    img_HR = cv2.cvtColor(img_HR, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_HR.shape
            # using matlab imresize
            img_LR = cv2.resize(img_HR, dsize=(int(H / scale), int(W / scale)), interpolation=cv2.INTER_CUBIC)
            img_D1 = cv2.resize(img_HR, dsize=(int(H / scale * 2), int(W / scale * 2)), interpolation=cv2.INTER_CUBIC)
            img_D2 = cv2.resize(img_HR, dsize=(int(H / scale * 4), int(W / scale * 4)), interpolation=cv2.INTER_CUBIC)
            img_D3 = cv2.resize(img_HR, dsize=(int(H / scale * 8), int(W / scale * 8)), interpolation=cv2.INTER_CUBIC)

            if img_LR.ndim == 2:
                img_LR = np.expand_dims(img_LR, axis=2)
                img_D1 = np.expand_dims(img_D1, axis=2)
                img_D2 = np.expand_dims(img_D2, axis=2)
                img_D3 = np.expand_dims(img_D3, axis=2)

        if self.opt['phase'] == 'train':
            # augmentation - flip, rotate
            img_LR, img_HR, img_D1, img_D2, img_D3 = util.augment([img_LR, img_HR, img_D1, img_D2, img_D3],
                                                                  self.opt['use_flip'], self.opt['use_rot'])
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HR.shape[2] == 3:
            img_HR = img_HR[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
            img_D1 = img_D1[:, :, [2, 1, 0]]
            img_D2 = img_D2[:, :, [2, 1, 0]]
            img_D3 = img_D3[:, :, [2, 1, 0]]
        img_HR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HR, (2, 0, 1)))).float()
        img_LR = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))).float()
        img_D1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D1, (2, 0, 1)))).float()
        img_D2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D2, (2, 0, 1)))).float()
        img_D3 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_D3, (2, 0, 1)))).float()

        if LR_path is None:
            LR_path = HR_path
        return {'LR': img_LR, 'HR': img_HR, 'LR_path': LR_path, 'HR_path': HR_path,
                'D1': img_D1, 'D2': img_D2, 'D3': img_D3, 'D1_path': D1_path, 'D2_path': D2_path, 'D3_path': D3_path}

    def __len__(self):
        return len(self.paths_HR)
