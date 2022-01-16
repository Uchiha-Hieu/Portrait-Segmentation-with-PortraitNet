import cv2
import torch

from .config import *
from .augmentation import *
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class dataset_1(Dataset):
    """
    dataset_1 : npy
    """
    def __init__(self,img_npy_path,mask_npy_path,split_ratio=1,isTrain = True,transform_train=True):
        img_npy_path = Path(img_npy_path)
        if img_npy_path.is_file() != True:
            raise ValueError("Path to image npy {} does not exist !".format(img_npy_path))
        mask_npy_path = Path(mask_npy_path)
        if mask_npy_path.is_file() != True:
            raise ValueError("Path to mask npy {} does not exist !".format(mask_npy_path))

        self.imgs = np.load(img_npy_path)
        self.masks = np.load(mask_npy_path)
        self.transform_train = transform_train
        # print(self.imgs.shape)
        if isTrain:
            num_train = int(self.imgs.shape[0]*split_ratio)
            self.imgs = self.imgs[:num_train]
            self.masks = self.masks[:num_train]

        else:
            num_train = int(self.imgs.shape[0]*split_ratio)
            #num_val = int(self.imgs.shape[0]*0.75)
            self.imgs = self.imgs[num_train:]
            self.masks = self.masks[num_train:]

        self.isTrain = isTrain

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        img = np.array(img,dtype=np.float32)

        if self.transform_train:
            img,mask = resize(img,mask)
            img_deformation,mask = randomHorizontalFlip(img,mask)
            img_deformation,mask = randomVerticalFlip(img_deformation,mask)
            img_texture = randomBlur(img_deformation)
            img_texture = randomHSV(img_texture)

            img_deformation = normalize_img(img_deformation)
            img_texture = normalize_img(img_texture)
            mask = normalize_mask(mask)
            mask_boundary = cv2.Canny(np.uint8(mask),0.3,0.5)

            img_texture = torch.from_numpy(np.transpose(img_texture,(2,0,1)).copy())
            img_deformation = torch.from_numpy(np.transpose(img_deformation,(2,0,1)).copy())
            mask = torch.from_numpy(mask.copy()).long()
            mask_boundary = (torch.from_numpy(mask_boundary.copy())/255.0).long()
            return img_deformation,img_texture,mask,mask_boundary

        else:
            img,mask = resize(img,mask)
            img = normalize_img(img)
            mask = normalize_mask(mask)
            mask_boundary = cv2.Canny(np.uint8(mask),0.3,0.5)
            img = torch.from_numpy(np.transpose(img,(2,0,1)).copy())
            mask = torch.from_numpy(mask.copy()).long()
            mask_boundary = (torch.from_numpy(mask_boundary.copy())/255.0).long()
            return img,mask,mask_boundary


       