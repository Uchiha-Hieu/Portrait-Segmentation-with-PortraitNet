import numpy as np
import cv2
from .config import *
import random
import copy


def resize(img, mask):
    img = cv2.resize(img, (IMG_NEW_WIDTH, IMG_NEW_HEIGHT), cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (IMG_NEW_WIDTH, IMG_NEW_HEIGHT), cv2.INTER_LINEAR)
    return img, mask

def normalize_img(img):
    img[:,:,0] = (img[:,:,0]/255.0 - MEAN[0])/STD[0]
    img[:,:,1] = (img[:,:,1]/255.0 - MEAN[1])/STD[1]
    img[:,:,2] = (img[:,:,2]/255.0 - MEAN[2])/STD[2]
    return img

def normalize_mask(mask):
    if np.max(mask) == 255.0:
        mask = mask/255.0
    return mask

def denormalize(img,mask):
    pass

def randomBlur(img):
    # print("RANDOMBLUR")
    if random.random() <= 0.5:
        img = cv2.blur(img, (3, 3))
    return img

def randomHSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    k = random.uniform(0.5, 1.5)
    if random.random() <= 0.5:
        # print("H")
        h = h * k
        h = np.clip(h, 0, 255).astype(hsv.dtype)

    if random.random() <= 0.5:
        # print("V")
        v = v * k
        v = np.clip(v, 0, 255).astype(hsv.dtype)

    if random.random() <= 0.5:
        # print("S")
        s = s * k
        s = np.clip(s, 0, 255).astype(hsv.dtype)

    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

def randomHorizontalFlip(img, mask):
    if random.random() <= 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    return img, mask

def randomVerticalFlip(img, mask):
    if random.random() <= 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)

    return img, mask
