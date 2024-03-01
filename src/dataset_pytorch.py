from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch
import cv2

from glob import glob
from tqdm import tqdm
from PIL import Image
from os import rename
from os import listdir
from shutil import move
from PIL import ImageFile
from os.path import splitext
import os

import re
import json

class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, human_annot, robot_annot, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.human_annot = human_annot
        self.robot_annot = robot_annot
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, mask_flag=False):
        
        img_nd = np.array(pil_img)
        img_nd = cv2.resize(img_nd, (256,256), interpolation = cv2.INTER_AREA)
        #img_nd = np.expand_dims(img_nd, axis=2)
        
        if mask_flag:
            msk = (img_nd > 100).astype(np.uint8)
            return msk.transpose((2, 0, 1))

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):

        img_file = self.ids[i]
        img = Image.open(self.imgs_dir + img_file)
        img = self.preprocess(img, self.scale, mask_flag = False)
        
        mask_img = Image.open(self.masks_dir + img_file)
        mask = self.preprocess(mask_img, self.scale, mask_flag=True)
        
        
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
    

    @staticmethod    
    def get_int(strx):
        return int(re.search(r'\d+', strx).group())


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')


class DatasetRetriever():

    def __init__(self, img_dir, mask_dir, batch_size, percentage_val, rapid_tests = False, num_workers=14):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.percentage_val = percentage_val
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.rapid_tests = rapid_tests

    def get_dataloaders(self):
        
        human_annot = self.__load_jsons_from_folder__(self.mask_dir + "human/")
        robot_annot = self.__load_jsons_from_folder__(self.mask_dir + "robot/")

        dataset = BasicDataset(self.img_dir, self.mask_dir, human_annot, robot_annot)
        
        n_val      = int(len(dataset) * self.percentage_val)
        n_train    = len(dataset) - n_val
        train, val = random_split(dataset, [n_train, n_val])
        
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
        val_loader   = DataLoader(val,   batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        if not self.rapid_tests:
            return (train_loader, val_loader)
        else:
            return (val_loader, val_loader)
