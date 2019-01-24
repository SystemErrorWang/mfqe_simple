#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:36:02 2019

@author: secret_wang
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data_utils import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from MC_subnet import MotionCompensateSubnet
from QE_subnet import QualityEnhanceSubnet
#from QP_network import QualityPredictNetwork
from tensorboardX import SummaryWriter



def train_qp_network():
    video_folder = 'dataset/compressed_small_x265'
    psnr_folder = 'dataset/npy_small_x265'
    total_iter = 0
    total_epochs = 30
    min_loss = 1e10
    writer = SummaryWriter('qp_log')
    model = QualityPredictNetwork().cuda()
    criterion = nn.MSELoss().cuda()
    '''
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    '''
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=-1)
    
    
    for epoch in range(total_epochs):
        if np.mod(epoch, 3) == 0:
            dataset = QPDataset(video_folder, psnr_folder, 
                                0, 8, transform=RandomCrop(512))
        elif np.mod(epoch, 3) == 1:
            dataset = QPDataset(video_folder, psnr_folder, 
                                8, 16, transform=RandomCrop(512))
        else:
            dataset = QPDataset(video_folder, psnr_folder, 
                                16, 24, transform=RandomCrop(512))
        dataloader = DataLoader(dataset, batch_size=32, 
                                shuffle=True, num_workers=0)
        scheduler.step()
        for idx, batch in tqdm(enumerate(dataloader)):
            total_iter += 1
            image = batch[0].float().cuda()
            psnr = batch[1].float().cuda()
            psnr_out = model(image)
            loss = criterion(psnr_out, psnr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('TrainLoss', loss.item(), total_iter)
            if np.mod(idx+1, 100) == 0:
                print('{}th epoch, {}th iteration, loss:{}'\
                      .format(epoch, idx, loss.item()))
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), 
                       'quality_predict_network.pth')
        dataset, dataloader = None, None
            
            
def train_joint():
    o_folder = 'dataset/original_videos'
    c_folder = 'dataset/compressed_small_x265'
    
    #transform = transforms.Compose([RandomCrop(512)])

    writer = SummaryWriter('log_joint')
    total_iter, mc_epochs, joint_epochs = 0, 2, 500
    min_loss_mc, min_loss = 1e10, 1e10
    mcnet = MotionCompensateSubnet().cuda()
    qenet = QualityEnhanceSubnet().cuda()
    mc_criterion = nn.MSELoss().cuda()
    #mc_criterion = nn.L1Loss().cuda()
    qe_criterion = nn.MSELoss().cuda()
    all_params = list(mcnet.parameters()) + list(qenet.parameters())
    mc_optimizer = torch.optim.Adam(mcnet.parameters(), 1e-4, weight_decay=5e-4)
    joint_optimizer = torch.optim.Adam(all_params, 1e-4, weight_decay=5e-4)

    dataset = SplitDataset(0, 25, transform=RandomCrop(128))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    for epoch in range(mc_epochs):
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            total_iter += 1
            before = batch[0].float().cuda()
            now = batch[1].float().cuda()
            after = batch[2].float().cuda()
            
            compensate1 = mcnet(now, before)
            compensate2 = mcnet(now, after)
            
            mc_loss = mc_criterion(compensate1, now) + mc_criterion(compensate2, now)
            mc_optimizer.zero_grad()
            mc_loss.backward()
            mc_optimizer.step()
            writer.add_scalar('mc_loss', mc_loss.item(), total_iter)
            #writer.add_scalar('qe_loss', qe_loss.item(), total_iter)
            #writer.add_scalar('total_loss', loss.item(), total_iter)
            
        print('{}th epoch, {}th iteration, loss:{}'\
                      .format(epoch, total_iter, mc_loss.item()))
        if mc_loss.item() < min_loss_mc:
            min_loss = mc_loss.item()
            torch.save(mcnet.state_dict(), 
                       'motion_compensate_network.pth')
        
    
    dataset, dataloader = None, None
    for epoch in range(joint_epochs):

        idx = np.mod(epoch, 25)
        dataset = SimpleDataset(o_folder, c_folder, idx, transform=RandomCrop(128))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            total_iter += 1
            c_before, c_after = batch[0].float().cuda(), batch[2].float().cuda()
            c_now, o_now = batch[1].float().cuda(), batch[3].float().cuda()
            
            compensate1 = mcnet(c_now, c_before)
            compensate2 = mcnet(c_now, c_after)
            enhance = qenet(compensate1, c_now, compensate2)
            
            mc_loss = mc_criterion(compensate1, c_now) + mc_criterion(compensate2, c_now)
            qe_loss = qe_criterion(enhance, o_now)
      
            loss = 1e-2*mc_loss + qe_loss
            joint_optimizer.zero_grad()
            loss.backward()
            joint_optimizer.step()
            writer.add_scalar('mc_loss', mc_loss.item(), total_iter)
            writer.add_scalar('qe_loss', qe_loss.item(), total_iter)
            writer.add_scalar('total_loss', loss.item(), total_iter)
            
        print('{}th epoch, {}th iteration, loss:{}'\
                      .format(epoch, total_iter, loss.item()))
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(mcnet.state_dict(), 
                       'motion_compensate_network.pth')
            torch.save(qenet.state_dict(), 
                       'quality_enhance_network.pth')
        dataloader, dataset = None, None
            

            
    
if __name__ == '__main__':
    train_joint()