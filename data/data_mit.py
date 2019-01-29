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
        o_before = cv2.cvtColor(o_before, cv2.COLOR_BGR2YUV)
        o_before = np.expand_dims(o_before[:, :, 0], 0)
        
        o_cap.set(1, 3)
        res, o_now = o_cap.read()
        o_now = cv2.cvtColor(o_now, cv2.COLOR_BGR2YUV)
        o_now = np.expand_dims(o_now[:, :, 0], 0)
        
        o_cap.set(1, 6)
        res, o_after = o_cap.read()
        o_after = cv2.cvtColor(o_after, cv2.COLOR_BGR2YUV)
        o_after = np.expand_dims(o_after[:, :, 0], 0)
        
        if self.transform:
            o_before = self.transform(o_before)
            o_now = self.transform(o_now)
            o_after = self.transform(o_after)
            
            
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
        try:
            o_now = cv2.cvtColor(o_now, cv2.COLOR_BGR2YUV)
        except:
            print(o_path)
        o_now = np.expand_dims(o_now[:, :, 0], 0)
        
        #c_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 0)
        c_cap.set(1, 0)
        res, c_before = c_cap.read()
        try:
            c_before = cv2.cvtColor(c_before, cv2.COLOR_BGR2YUV)
        except:
            print(c_path)
        c_before = np.expand_dims(c_before[:, :, 0], 0)
        
        #c_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 3)
        c_cap.set(1, 3)
        res, c_now = c_cap.read()
        try:
            c_now = cv2.cvtColor(c_now, cv2.COLOR_BGR2YUV)
        except:
            print(c_path)
        c_now = np.expand_dims(c_now[:, :, 0], 0)
        
        #c_cap.set(cv2.CV_CAP_PROP_POS_FRAMES, 6)
        c_cap.set(1, 6)
        res, c_after = c_cap.read()
        try:
            c_after = cv2.cvtColor(c_after, cv2.COLOR_BGR2YUV)
        except:
            print(c_path)
        c_after = np.expand_dims(c_after[:, :, 0], 0)
        
        if self.transform:
            o_now = self.transform(o_now)
            c_before = self.transform(c_before)
            c_now = self.transform(c_now)
            c_after = self.transform(c_after)
            
            
        return o_now/255.0, c_before/255.0, c_now/255.0, c_after/255.0

'''

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
        o_video = skvideo.io.vread(o_path)
        c_path = os.path.join(self.c_folder, self.c_list[index])
        c_video = skvideo.io.vread(c_path)
        
        #print(np.shape(o_video), np.shape(c_video))
        
        o_now = cv2.cvtColor(o_video[3], cv2.COLOR_BGR2YUV)
        o_now = np.expand_dims(o_now[:, :, 0], 0)
        
        c_before = cv2.cvtColor(c_video[0], cv2.COLOR_BGR2YUV)
        c_before = np.expand_dims(c_before[:, :, 0], 0)
        
        c_now = cv2.cvtColor(c_video[3], cv2.COLOR_BGR2YUV)
        c_now = np.expand_dims(c_now[:, :, 0], 0)
        
        c_after = cv2.cvtColor(c_video[6], cv2.COLOR_BGR2YUV)
        c_after = np.expand_dims(c_after[:, :, 0], 0)
        
        if self.transform:
            o_now = self.transform(o_now)
            c_before = self.transform(c_before)
            c_now = self.transform(c_now)
            c_after = self.transform(c_after)
            
            
        return o_now/255.0, c_before/255.0, c_now/255.0, c_after/255.0
'''

    
class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[1:]
        new_h, new_w = self.output_size
        try:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            image = image[:, top: top + new_h,
                      left: left + new_w]
        except:
            print(h, w)
            print(self.output_size)
        
        return image


class RandomFlip(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        pass

    
'''
o_folder = 'C:\\Users\\Administrator\\Downloads\\mit_dataset'
c_folder = 'C:\\Users\\Administrator\\Downloads\\mit_compress'

dataset = JointDataset(o_folder, c_folder, transform=RandomCrop(128))
#dataset = MCDataset(o_folder, transform=RandomCrop(128))
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
for batch in tqdm(dataloader, total=len(dataloader)):
    pass
'''


