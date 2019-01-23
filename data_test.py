#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:40:20 2019

@author: secret_wang
"""
import os
import cv2
import torch
import time
import numpy as np
import skvideo
skvideo.setFFmpegPath("C:/Program Files/ffmpeg/bin")
import skvideo.io
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



class QPDataset(Dataset):
    def __init__(self, video_folder, psnr_folder, 
                 start, end, transform=None):
        self.psnr_files = []
        self.video_files = []
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for idx in range(start, end):
            timer1 = time.time()
            y_video = []
            video_path = os.path.join(video_folder, 
                        'compressed_x265_small_{}.mov'.format(idx))
            psnr_path = os.path.join(psnr_folder, 
                                     'psnr_{}.npy'.format(idx))
            cap = cv2.VideoCapture(video_path)
            while (cap.isOpened()):
                res, frame = cap.read()
                if type(frame) == type(None):
                    break
                frame_y = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:, :, 0]
                frame_y = np.expand_dims(frame_y, 0)
                y_video.append(frame_y)
            
            
            length = len(y_video)
            self.video_files.append(y_video)
            self.psnr_files.append(np.load(psnr_path))
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            print('load a video, time:{}'.format(time.time()-timer1))
            #print(np.shape(y_video))
    
        
    def __len__(self):
        return np.sum(self.all_length)
  
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        psnr = self.psnr_files[video_idx][frame_idx]
        frame = self.video_files[video_idx][frame_idx]
        
        if self.transform:
            frame = self.transform(frame)
            
        return frame, psnr
        

    

class JointDataset(Dataset):

    def __init__(self, o_folder, c_folder, psnr_folder, 
                 num_videos=24, transform=None):
        self.o_folder = o_folder
        self.c_folder = c_folder
        self.psnr_folder = psnr_folder
        self.num_videos = num_videos
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for idx in range(num_videos):
            video_path = os.path.join(o_folder, 
                        'original_{}.mov'.format(idx))
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        
        o_path = os.path.join(self.o_folder, 
                    'original_{}.mov'.format(video_idx))
        c_path = os.path.join(self.c_folder, 
                    'compressed_x265_small_{}.mov'.format(video_idx))
        cap = cv2.VideoCapture(c_path)
        cap_hr = cv2.VideoCapture(o_path)
        
        psnr_path = os.path.join(self.psnr_folder, 
                    'psnr_{}.npy'.format(video_idx))
        psnr_file = np.load(psnr_path)
        
        idx_before, idx_after = find_pqf(psnr_file, frame_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_before)
        res, frame_before = cap.read()  
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, idx_before)
        res, hr_before = cap_hr.read()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, frame_now = cap.read()  
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, hr_now = cap_hr.read()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_after)
        res, frame_after = cap.read()  
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, idx_after)
        res, hr_after = cap_hr.read()
        
        if self.transform:
            frame_before, hr_before = self.transform(frame_before), self.transform(hr_before)
            frame_now, hr_now = self.transform(frame_now), self.transform(hr_now)
            frame_after, hr_after = self.transform(frame_after), self.transform(hr_after)
            
        return frame_before, hr_before, frame_now, hr_now, frame_after, hr_after


class MCDataset(Dataset):

    def __init__(self, o_folder, psnr_folder, 
                 num_videos=24, transform=None):
        self.o_folder = o_folder
        self.psnr_folder = psnr_folder
        self.num_videos = num_videos
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for idx in range(num_videos):
            video_path = os.path.join(o_folder, 
                        'original_{}.mov'.format(idx))
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        
        o_path = os.path.join(self.o_folder, 
                    'original_{}.mov'.format(video_idx))
        cap_hr = cv2.VideoCapture(o_path)
        
        psnr_path = os.path.join(self.psnr_folder, 
                    'psnr_{}.npy'.format(video_idx))
        psnr_file = np.load(psnr_path)
        
        idx_before, idx_after = find_pqf(psnr_file, frame_idx)
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, idx_before)
        res, hr_before = cap_hr.read()
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, hr_now = cap_hr.read()
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, idx_after)
        res, hr_after = cap_hr.read()
        
        if self.transform:
            hr_before = self.transform(hr_before)
            hr_now = self.transform(hr_now)
            hr_after = self.transform(hr_after)
            
        return hr_before, hr_now, hr_after
    
    
class QEDataset(Dataset):

    def __init__(self, o_folder, c_folder, psnr_folder, 
                 num_videos=24, transform=None):
        self.o_folder = o_folder
        self.c_folder = c_folder
        self.psnr_folder = psnr_folder
        self.num_videos = num_videos
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for idx in range(num_videos):
            video_path = os.path.join(o_folder, 
                        'original_{}.mov'.format(idx))
            cap = cv2.VideoCapture(video_path)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        
        o_path = os.path.join(self.o_folder, 
                    'original_{}.mov'.format(video_idx))
        c_path = os.path.join(self.c_folder, 
                    'compressed_x265_small_{}.mov'.format(video_idx))
        cap = cv2.VideoCapture(c_path)
        cap_hr = cv2.VideoCapture(o_path)
        
        psnr_path = os.path.join(self.psnr_folder, 
                    'psnr_{}.npy'.format(video_idx))
        psnr_file = np.load(psnr_path)
        
        idx_before, idx_after = find_pqf(psnr_file, frame_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_before)
        res, frame_before = cap.read()  
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, frame_now = cap.read()  
        
        cap_hr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        res, hr_now = cap_hr.read()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_after)
        res, frame_after = cap.read()  
        
        
        if self.transform:
            frame_before, frame_after = self.transform(frame_before), self.transform(frame_after)
            frame_now, hr_now = self.transform(frame_now), self.transform(hr_now)
           
            
        return frame_before, frame_now, frame_after, hr_now
    
    
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
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[:, top: top + new_h,
                      left: left + new_w]
        return image


class ToYUV(object):

    def __init__(self):
        pass

    def __call__(self, image):
        y_channel = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)[:, :, 0]
        y_channel = np.expand_dims(y_channel, 0)
        return y_channel
    
    

video_folder = 'dataset/compressed_small_x265'
psnr_folder = 'dataset/npy_small_x265'
'''
dataset = QPDataset(video_folder, psnr_folder, 0, 8, transform=RandomCrop(512))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
'''
for epoch in range(3):
    if np.mod(epoch, 3) == 0:
        #dataset = None
        dataset = QPDataset(video_folder, psnr_folder, 
                            0, 8, transform=RandomCrop(512))
    elif np.mod(epoch, 3) == 1:
        #dataset = None
        dataset = QPDataset(video_folder, psnr_folder, 
                            8, 16, transform=RandomCrop(512))
    else:
        #dataset = None
        dataset = QPDataset(video_folder, psnr_folder, 
                            16, 24, transform=RandomCrop(512))
    dataloader = DataLoader(dataset, batch_size=32, 
                            shuffle=True, num_workers=0)
    for batch in tqdm(dataloader):
        pass
    dataset, dataloader = None, None

