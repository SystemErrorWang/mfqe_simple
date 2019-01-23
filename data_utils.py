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

    def __init__(self, o_folder, c_folder, 
                 start, end, transform=None):
        self.o_files = []
        self.c_files = []
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for idx in range(start, end):
            timer1 = time.time()
            o_frames, c_frames = [], []
            o_path = os.path.join(o_folder, 
                        'original_{}.mov'.format(idx))
            c_path = os.path.join(c_folder, 
                        'compressed_x265_small_{}.mov'.format(idx))
       
            o_cap = cv2.VideoCapture(o_path)
            c_cap = cv2.VideoCapture(c_path)
            while (o_cap.isOpened() and c_cap.isOpened()):
                res, o_frame = o_cap.read()
                res, c_frame = c_cap.read()
                if type(o_frame) == type(None) or type(c_frame) == type(None):
                    break
                o_y = cv2.cvtColor(o_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
                o_y = np.expand_dims(o_y, 0)
                c_y = cv2.cvtColor(c_frame, cv2.COLOR_BGR2YUV)[:, :, 0]
                c_y = np.expand_dims(c_y, 0)
                o_frames.append(o_y)
                c_frames.append(c_y)
            
            
            length = len(o_frames)
            self.c_files.append(c_frames)
            self.o_files.append(o_frames)
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            print('load a video, time:{}'.format(time.time()-timer1))
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        o_now = self.o_files[video_idx][frame_idx]
        c_now = self.c_files[video_idx][frame_idx]
        if frame_idx <= 0:
            c_before = c_now
            c_after = self.c_files[video_idx][frame_idx+1]
        elif frame_idx >= self.all_length[video_idx]-1:
            c_before = self.c_files[video_idx][frame_idx-1]
            c_after = c_now
        else:
            c_before = self.c_files[video_idx][frame_idx-1]
            c_after = self.c_files[video_idx][frame_idx+1]
    
        if self.transform:
            c_before, c_after = self.transform(c_before), self.transform(c_after)
            c_now, o_now = self.transform(c_now), self.transform(o_now)
            
        return c_before, c_now, c_after, o_now
    
    
    
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


class RandomFlip(object):
    def __init__(self):
        pass
    
    def __call__(self, image):
        pass

    

o_folder = 'dataset/original_videos'
c_folder = 'dataset/compressed_small_x265'

for epoch in range(4):
    if np.mod(epoch, 4) == 0:
        dataset = JointDataset(o_folder, c_folder, 0, 6, transform=RandomCrop(512))
    elif np.mod(epoch, 4) == 1:
        dataset = JointDataset(o_folder, c_folder, 6, 12, transform=RandomCrop(512))
    elif np.mod(epoch, 4) == 2:
        dataset = JointDataset(o_folder, c_folder, 12, 18, transform=RandomCrop(512))
    elif np.mod(epoch, 4) == 3:
        dataset = JointDataset(o_folder, c_folder, 18, 24, transform=RandomCrop(512))
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    for batch in tqdm(dataloader, total=len(dataloader)):
        pass
    dataset, dataloader = None, None

