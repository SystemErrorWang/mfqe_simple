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
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def find_idx(idx, len_sep):
    start, end = 0, len(len_sep)-1
    while start <= end:
        mid = start + (end - start) // 2
        if len_sep[mid] < idx:
            start = mid + 1
        elif len_sep[mid] > idx:
            end = mid - 1
        else:
            out1, out2 = mid, idx - len_sep[mid]
            if out2 < 0:
                out1 = out1 - 1
                out2 = idx - len_sep[out1]
            return out1, out2
        
    out1, out2 = mid, idx - len_sep[mid]
    if out2 < 0:
        out1 = out1 - 1
        out2 = idx - len_sep[out1]
    return out1, out2



def find_pqf(psnr, idx):
    length = len(psnr)
    psnr_before = psnr[max(0, idx-5): idx]
    psnr_after = psnr[idx+1: min(length+1, idx+6)]
    
    if len(psnr_before) > 0:
        pb_max = np.max(psnr_before)
        pb_idx = np.where(psnr_before == pb_max)[0][0]
    else:
        pb_max, pb_idx = -1, 0
        
    if len(psnr_after) > 0:
        pa_max = np.max(psnr_after)
        pa_idx = np.where(psnr_after == pa_max)[0][0]
  
    else:
        pa_max, pa_idx = -1, 0
    
    if pb_max > psnr[idx] and pa_max > psnr[idx]:
        idx_before = pb_idx + idx - 1 -len(psnr_before)
        idx_after = pa_idx + idx
        return idx_before, idx_after
    else: 
        return idx, idx
    


class MCDataset(Dataset):
    def __init__(self, o_folder, transform=None):
        
        self.all_length = []
        self.len_sep = [0]
        self.o_files = []
        self.transform = transform
        for folder in os.listdir(o_folder):
            o_paths = []
            o_subfolder = os.path.join(o_folder, folder)
       
            for name in os.listdir(o_subfolder):
                o_path = os.path.join(o_subfolder, name)
                o_paths.append(o_path)
                
            length = len(o_paths)
            self.o_files.append(o_paths)
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        sub_len = self.all_length[video_idx]-1

        o_now = self.o_files[video_idx][frame_idx]
        o_before = self.o_files[video_idx][max(frame_idx-3, 0)]
        o_after = self.o_files[video_idx][min(frame_idx+3, sub_len)]
        
        o_now = cv2.cvtColor(cv2.imread(o_now), cv2.COLOR_BGR2YUV)
        o_now = np.expand_dims(o_now[:, :, 0], 0)
        o_before = cv2.cvtColor(cv2.imread(o_before), cv2.COLOR_BGR2YUV)
        o_before = np.expand_dims(o_before[:, :, 0], 0)
        o_after = cv2.cvtColor(cv2.imread(o_after), cv2.COLOR_BGR2YUV)
        o_after = np.expand_dims(o_after[:, :, 0], 0)
    
        if self.transform:
            o_before, o_now, o_after = self.transform(o_before, o_now, o_after)
        
        o_before = o_before.astype(np.float32)/255.0
        o_now = o_now.astype(np.float32)/255.0
        o_after = o_after.astype(np.float32)/255.0
            
        return o_before, o_now, o_after
    
        

class JointDataset(Dataset):

    def __init__(self, o_folder, c_folder, transform=None):
        
        self.all_length = []
        self.len_sep = [0]
        self.o_files = []
        self.c_files = []
        self.transform = transform
        for folder in os.listdir(o_folder):
            
            o_paths, c_paths = [], []
            o_subfolder = os.path.join(o_folder, folder)
            c_subfolder = os.path.join(c_folder, folder)
       
            for name in os.listdir(o_subfolder):
                o_path = os.path.join(o_subfolder, name)
                c_path = os.path.join(c_subfolder, name)
                o_paths.append(o_path)
                c_paths.append(c_path)
                
            length = len(o_paths)
            self.c_files.append(c_paths)
            self.o_files.append(o_paths)
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        sub_len = self.all_length[video_idx]-1
        o_now = self.o_files[video_idx][frame_idx]
        c_now = self.c_files[video_idx][frame_idx]
        c_before = self.c_files[video_idx][max(frame_idx-3, 0)]
        c_after = self.c_files[video_idx][min(frame_idx+3, sub_len)]
        
        o_now = cv2.cvtColor(cv2.imread(o_now), cv2.COLOR_BGR2YUV)
        o_now = np.expand_dims(o_now[:, :, 0], 0)
        c_now = cv2.cvtColor(cv2.imread(c_now), cv2.COLOR_BGR2YUV)
        c_now = np.expand_dims(c_now[:, :, 0], 0)
        c_before = cv2.cvtColor(cv2.imread(c_before), cv2.COLOR_BGR2YUV)
        c_before = np.expand_dims(c_before[:, :, 0], 0)
        c_after = cv2.cvtColor(cv2.imread(c_after), cv2.COLOR_BGR2YUV)
        c_after = np.expand_dims(c_after[:, :, 0], 0)
    
        if self.transform:
            o_now, c_before, c_now, c_after = self.transform(o_now, c_before, c_now, c_after)
        
        c_before = c_before.astype(np.float32)/255.0
        c_now = c_now.astype(np.float32)/255.0
        c_after = c_after.astype(np.float32)/255.0
        o_now = o_now.astype(np.float32)/255.0
            
        return o_now, c_before, c_now, c_after
    

    

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
    

'''
o_folder = 'C:\\Users\\Administrator\\Downloads\\davis\\hr_image'
c_folder = 'C:\\Users\\Administrator\\Downloads\\davis\\lr_image'

dataset = JointDataset(o_folder, c_folder, transform=CropFourFrames(512))
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
idx = 0
for batch in tqdm(dataloader, total=len(dataloader)):
    o, _, c, _ = batch[:]
    _, _, h, w = o.size()
    o = o.reshape(h, w, 1)*255.0
    c = c.reshape(h, w, 1)*255.0
    o = o.numpy().astype(np.uint8)
    c = c.numpy().astype(np.uint8)
    cv2.imwrite('{}hr.jpg'.format(idx), o)
    cv2.imwrite('{}lr.jpg'.format(idx), c)
    idx += 1
'''
