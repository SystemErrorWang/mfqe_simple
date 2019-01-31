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
    def __init__(self, video_folder, 
                 start, end, transform=None):
        self.video_files = []
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for idx in range(start, end):
            timer1 = time.time()
            y_video = []
            video_path = os.path.join(video_folder, 
                        'original_{}.mov'.format(idx))
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
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            print('load a video, time:{}'.format(time.time()-timer1))
    
        
    def __len__(self):
        return np.sum(self.all_length)
  
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        frame_now = self.video_files[video_idx][frame_idx]
        if frame_idx <= 0:
            frame_before = frame_now
            frame_after = self.video_files[video_idx][frame_idx+1]
        elif frame_idx >= self.all_length[video_idx]-1:
            frame_before = self.video_files[video_idx][frame_idx-1]
            frame_after = frame_now
        else:
            frame_before = self.video_files[video_idx][frame_idx-1]
            frame_after = self.video_files[video_idx][frame_idx+1]
        if self.transform:
            frame_before = self.transform(frame_before)
            frame_now = self.transform(frame_now)
            frame_after = self.transform(frame_after)
            
        return frame_before/255.0, frame_now/255.0, frame_after/255.0     
        

class JointDataset(Dataset):

    def __init__(self, o_folder, c_folder, transform=None):
        self.o_files = []
        self.c_files = []
        self.all_length = []
        self.len_sep = [0]
        self.transform = transform
        for name in tqdm(os.listdir(o_folder)):
            
            o_frames, c_frames = [], []
            o_path = os.path.join(o_folder, name)
            c_path = os.path.join(c_folder, name)
       
            o_cap = cv2.VideoCapture(o_path)
            c_cap = cv2.VideoCapture(c_path)
            while (o_cap.isOpened() and c_cap.isOpened()):
                res, o_frame = o_cap.read()
                res, c_frame = c_cap.read()
                if type(o_frame) == type(None) or type(c_frame) == type(None):
                    break
                o_y, _, _ = bgr2yuv(o_frame)
                o_y = np.expand_dims(o_y, 0)
                c_y, _, _ = bgr2yuv(c_frame)
                c_y = np.expand_dims(c_y, 0)
                o_frames.append(o_y)
                c_frames.append(c_y)
            
            length = len(o_frames)
            self.c_files.append(c_frames)
            self.o_files.append(o_frames)
            self.all_length.append(length)
            self.len_sep.append(np.sum(self.all_length))
            
            
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        video_len = self.all_length[video_idx]-1
        o_now = self.o_files[video_idx][frame_idx]
        c_now = self.c_files[video_idx][frame_idx]
        c_before = self.c_files[video_idx][max(frame_idx-3, 0)]
        c_after = self.c_files[video_idx][min(frame_idx+3, video_len)]
    
        if self.transform:
            o_now, c_before, c_now, c_after = self.transform(o_now, c_before, c_now, c_after)
            
        return c_before/255.0, c_now/255.0, c_after/255.0, o_now/255.0
    
    
class SimpleDataset(Dataset):
    '''
    output should be (bs, 1, h, w)
    image in y channel rescaled to (0, 1) 
    '''

    def __init__(self, o_folder, c_folder, transform=None):
        self.all_length = []
        self.len_sep = [0]
        self.o_path = []
        self.c_path = []
        self.transform = transform
        
        for name in os.listdir(o_folder):
            o_path = os.path.join(o_folder, name)
            c_path = os.path.join(c_folder, name)
            self.o_path.append(o_path)
            self.c_path.append(c_path)
            cap = cv2.VideoCapture(o_path)
            frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.all_length.append(frame)
            self.len_sep.append(np.sum(self.all_length))
        
            
    def __len__(self):
        return np.sum(self.all_length)
        
            
    def __getitem__(self, index):
        video_idx, frame_idx = find_idx(index, self.len_sep)
        video_len = self.all_length[video_idx]-1
        
        o_cap = cv2.VideoCapture(self.o_path[video_idx])
        c_cap = cv2.VideoCapture(self.c_path[video_idx])
        
        try:
            o_cap.set(1, frame_idx)
            res, o_now = o_cap.read()
            #print(np.shape(o_now))
            o_now, _, _ = bgr2yuv(o_now)
            o_now = np.expand_dims(o_now, 0)
        except:
            print(self.o_path[video_idx])
        
        try:
            c_cap.set(1, max(frame_idx-3, 0))
            res, c_before = c_cap.read()
            #print(np.shape(c_before))
            c_before, _, _ = bgr2yuv(c_before)
            c_before = np.expand_dims(c_before, 0)
            
            c_cap.set(1, frame_idx)
            res, c_now = c_cap.read()
            #print(np.shape(c_now))
            c_now, _, _ = bgr2yuv(c_now)
            c_now = np.expand_dims(c_now, 0)
            
            c_cap.set(1, min(frame_idx+3, video_len))
            res, c_after = c_cap.read()
            #print(np.shape(c_after))
            c_after, _, _ = bgr2yuv(c_after)
            c_after = np.expand_dims(c_after, 0)
        except:
            print(self.c_path[video_idx])
        
        
        if self.transform:
            o_now, c_before, c_now, c_after = self.transform(o_now, c_before, c_now, c_after)
        
        return c_before/255.0, c_now/255.0, c_after/255.0, o_now/255.0    
    
    

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
    

o_folder = 'C:\\Users\\Administrator\\Downloads\\davis_18'
c_folder = 'C:\\Users\\Administrator\\Downloads\\davis_43'

dataset = JointDataset(o_folder, c_folder, transform=CropFourFrames(256))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
for batch in tqdm(dataloader, total=len(dataloader)):
    pass


