#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:36:02 2019

@author: secret_wang
"""

import torch
import os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from data.data_mit import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from network.MC_subnet import MotionCompensateSubnet
from network.QE_subnet import QualityEnhanceSubnet
#from QP_network import QualityPredictNetwork
from tensorboardX import SummaryWriter



def train_mc_network():
    o_folder = 'D:\\mit_dataset'
    c_folder = 'D:\\mit_compress'
    total_iter = 0
    total_epochs = 30
    min_loss = 1e10
    writer = SummaryWriter('mc_log')
    model = MotionCompensateSubnet().cuda()
    criterion = nn.MSELoss().cuda()
    '''
    optimizer = torch.optim.SGD(model.parameters(), 1e-4, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    '''
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95, last_epoch=-1)
    
    
    for epoch in range(total_epochs):
        dataset = MCDataset(o_folder, transform=CropThreeFrames(256))
        dataloader = DataLoader(dataset, batch_size=32, 
                                shuffle=True, num_workers=0)
        scheduler.step()
        for idx, batch in tqdm(enumerate(dataloader)):
            total_iter += 1
            frame_before = batch[0].float().cuda()
            frame_now = batch[1].float().cuda()
            frame_after = batch[2].float().cuda()
            
            mc_before = model(frame_now, frame_before)
            mc_after = model(frame_now, frame_after)
  
            loss = criterion(mc_before, frame_now) + criterion(mc_after, frame_now)
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
                       'weight\\motion_compensate_network.pth')
        dataset, dataloader = None, None
            
            
def train_joint():
    o_folder = 'C:\\Users\\Administrator\\Downloads\\mit_compress'
    c_folder = 'C:\\Users\\Administrator\\Downloads\\mit_compress_small'

    writer = SummaryWriter('log_joint')
    total_iter, mc_epochs, joint_epochs = 0, 1, 10
    min_loss_mc, min_loss = 1e10, 1e10
    mcnet = MotionCompensateSubnet().cuda()
    qenet = QualityEnhanceSubnet().cuda()
    mc_criterion = nn.MSELoss().cuda()
    #mc_criterion = nn.L1Loss().cuda()
    qe_criterion = nn.MSELoss().cuda()
    all_params = list(mcnet.parameters()) + list(qenet.parameters())
    mc_optimizer = torch.optim.Adam(mcnet.parameters(), 1e-4, weight_decay=5e-4)
    joint_optimizer = torch.optim.Adam(all_params, 1e-4, weight_decay=5e-4)

    dataset = MCDataset(o_folder, transform=CropThreeFrames(224))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
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
                       'weight\\mcnet_sep_{}th_epoch.pth'.format(epoch))
        
    
    dataset, dataloader = None, None
    for epoch in range(joint_epochs):
       
        dataset = JointDataset(o_folder, c_folder, transform=CropFourFrames(128))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            total_iter += 1
            ref = batch[0].float().cuda()
            before = batch[1].float().cuda()
            now = batch[2].float().cuda()
            after = batch[3].float().cuda()
            
            
            compensate1 = mcnet(now, before)
            compensate2 = mcnet(now, after)
            enhance = qenet(compensate1, now, compensate2)
            
            mc_loss = mc_criterion(compensate1, now) + mc_criterion(compensate2, now)
            qe_loss = qe_criterion(enhance, ref)
      
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
                       'weight\\mcnet_joint_{}th_epoch.pth'.format(epoch))
            torch.save(qenet.state_dict(), 
                       'weight\\qenet_joint_{}th_epoch.pth'.format(epoch))

            
    
if __name__ == '__main__':
    if not os.path.exists('weight'):
        os.mkdir('weight')
    train_joint()