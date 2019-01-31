#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:40:20 2019

@author: secret_wang
"""
import os
import cv2
import time
import torch
import numpy as np
import skvideo
skvideo.setFFmpegPath("C:/Program Files/ffmpeg/bin")
import skvideo.io
from tqdm import tqdm
from color_convert import bgr2yuv, yuv2bgr
from torch.utils.data import Dataset, DataLoader



class MCDataset(Dataset):
    def __init__(self, o_folder, transform=None):
        self.o_folder = o_folder
        self.o_list = os.listdir(o_folder)
        self.transform = transform
    
        
    def __len__(self):
        return len(self.o_list)
  
            
    def __getitem__(self, index):
        o_path = os.path.join(self.o_folder, self.o_list[index])
        o_cap = cv2.VideoCapture(o_path)
        
        o_cap.set(1, 0)
        res, o_before = o_cap.read()
        o_before, _, _ = bgr2yuv(o_before)
        o_before = np.expand_dims(o_before, 0)
        
        o_cap.set(1, 3)
        res, o_now = o_cap.read()
        o_now, _, _ = bgr2yuv(o_now)
        o_now = np.expand_dims(o_now, 0)
        
        o_cap.set(1, 6)
        res, o_after = o_cap.read()
        o_after, _, _ = bgr2yuv(o_after)
        o_after = np.expand_dims(o_after, 0)
        
        if self.transform:
            o_before, o_now, o_after = self.transform(o_before, o_now, o_after)
            
            
        return o_before/255.0, o_now/255.0, o_after/255.0



class JointDataset(Dataset):
    def __init__(self, o_folder, c_folder, transform=None):
        self.o_folder = o_folder
        self.o_list = os.listdir(o_folder)
        self.c_folder = c_folder
        self.c_list = os.listdir(c_folder)
        self.transform = transform
        
        
    def __len__(self):
        return len(self.o_list)
  
            
    def __getitem__(self, index):
        o_path = os.path.join(self.o_folder, self.o_list[index])
        o_cap = cv2.VideoCapture(o_path)
        c_path = os.path.join(self.c_folder, self.c_list[index])
        c_cap = cv2.VideoCapture(c_path)
        
        #o_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 3)
        o_cap.set(1, 3)
        res, o_now = o_cap.read()
        o_now, _, _ = bgr2yuv(o_now)
        o_now = np.expand_dims(o_now, 0)
        
        #c_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 0)
        c_cap.set(1, 0)
        res, c_before = c_cap.read()
        c_before, _, _ = bgr2yuv(c_before)
        c_before = np.expand_dims(c_before, 0)
        
        #c_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 3)
        c_cap.set(1, 3)
        res, c_now = c_cap.read()
        c_now, _, _ = bgr2yuv(c_now)
        c_now = np.expand_dims(c_now, 0)
        
        #c_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 6)
        c_cap.set(1, 6)
        res, c_after = c_cap.read()
        c_after, _, _ = bgr2yuv(c_after)
        c_after = np.expand_dims(c_after, 0)
        
        if self.transform:
            o_now, c_before, c_now, c_after = self.transform(o_now, c_before, 
                                                             c_now, c_after)
            
        return o_now/255.0, c_before/255.0, c_now/255.0, c_after/255.0



class CropThreeFrames(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img1, img2, img3):
        h, w = img1.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img1 = img1[:, top: top + new_h,
                  left: left + new_w]
        img2 = img2[:, top: top + new_h,
                  left: left + new_w]
        img3 = img3[:, top: top + new_h,
                  left: left + new_w]
       
        return img1, img2, img3


    
class CropFourFrames(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img1, img2, img3, img4):
        h, w = img1.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img1 = img1[:, top: top + new_h,
                  left: left + new_w]
        img2 = img2[:, top: top + new_h,
                  left: left + new_w]
        img3 = img3[:, top: top + new_h,
                  left: left + new_w]
        img4 = img4[:, top: top + new_h,
                  left: left + new_w]
        
        return img1, img2, img3, img4


class RandomFlip(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        pass

    
'''
o_folder = 'C:\\Users\\Administrator\\Downloads\\mit_dataset'
c_folder = 'C:\\Users\\Administrator\\Downloads\\mit_compress'

dataset = JointDataset(o_folder, c_folder, transform=CropFourFrames(128))
#dataset = MCDataset(o_folder, transform=RandomCrop(128))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
for batch in tqdm(dataloader, total=len(dataloader)):
    pass
'''



