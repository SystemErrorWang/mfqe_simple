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
from QP_network import QualityPredictNetwork
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
    
    transform = transforms.Compose([RandomCrop(512)])

    writer = SummaryWriter('log_joing')
    total_iter, total_epochs = 0, 160
    min_loss_mc, min_loss_qe = 1e10, 1e10
    mcnet = MotionCompensateSubnet().cuda()
    qenet = QualityEnhanceSubnet().cuda()
    mc_criterion = nn.MSELoss().cuda()
    qe_criterion = nn.MSELoss().cuda()
    all_params = list(model1.parameters()) + list(model2.parameters())
    optimizer = torch.optim.SGD(all_params, 2e-4, momentum=0.9,
                                   weight_decay=5e-4, nesterov=True)
    
    for epoch in range(mc_epochs):
        if np.mod(epoch, 4) == 0:
            dataset = JointDataset(o_folder, c_folder, 0, 6, transform=RandomCrop(512))
        elif np.mod(epoch, 4) == 1:
            dataset = JointDataset(o_folder, c_folder, 6, 12, transform=RandomCrop(512))
        elif np.mod(epoch, 4) == 2:
            dataset = JointDataset(o_folder, c_folder, 12, 18, transform=RandomCrop(512))
        elif np.mod(epoch, 4) == 3:
            dataset = JointDataset(o_folder, c_folder, 18, 24, transform=RandomCrop(512))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
        for idx, batch in enumerate(dataloader):
            total_iter += 1
            c_before, c_after = batch[0].cuda(), batch[2].cuda()
            c_now, o_now = batch[1].cuda(), batch[3].cuda()
            
            compensate1 = mcnet(c_now, c_before)
            compensate2 = mcnet(c_now, c_after)
            enhance = qenet(compensate1, compensate2)
            
            mc_loss = mc_criterion(compensate1, o_now) + mc_criterion(compensate2, o_now)
            qe_loss = qe_criterion(enhance, o_now)
            if epoch < 80:
                loss = mc_loss + 1e-2*qe_loss
            else:
                loss = 1e-2*mc_loss + qe_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('mc_loss', mc_loss.item(), total_iter)
            writer.add_scalar('qe_loss', qe_loss.item(), total_iter)
            writer.add_scalar('total_loss', loss.item(), total_iter)
            if np.mod(idx+1, 100) == 0:
                print('{}th epoch, {}th iteration, loss:{}'\
                      .format(epoch, total_iter, mc_loss.item()))
        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(mcnet.state_dict(), 
                       'motion_compensate_network.pth')
            torch.save(qenet.state_dict(), 
                       'quality_enhance_network.pth')
        dataloader, dataset = None, None
            
    '''
    qe_writer = SummaryWriter('qe_log')
    mcnet.load_state_dict('motion_compensate_network.pth')
    qe_dataset = QEDataset(o_folder, psnr_folder, transform=transform)
    qe_dataloader = DataLoader(qe_dataset, batch_size=8, shuffle=True, num_workers=4)
    for epoch in range(qe_epochs):
        for idx, batch in enumerate(qe_dataloader):
            c_before, c_now = batch[0].cuda(), batch[1].cuda()
            c_after, o_now = batch[2].cuda(), batch[3].cuda()
            
            compensate1 = mcnet(c_now, c_before)
            compensate2 = mcnet(c_now, c_after)
            enhance = qenet(compensate1, compensate2)
            
            mc_loss = mc_criterion(compensate1, c_now) + mc_criterion(compensate2, c_now)
            qe_loss = qe_criterion(enhance, o_now)
            total_loss = 1e-3 * mc_loss + qe_loss
            qe_optimizer.zero_grad()
            total_loss.backward()
            qe_optimizer.step()
            qe_writer.add_scalar('qe_loss', qe_loss.item(), 
                              epoch*len(qe_dataloader)+idx)
            if np.mod(idx+1, 100) == 0:
                print('{}th epoch, {}th iteration, loss:{}'\
                      .format(epoch, idx, total_loss.item()))
                
        if total_loss.item() < min_loss_qe:
            min_loss_qe = total_loss.item()
            torch.save(qenet.state_dict(), 
                       'quality_enhance_network.pth')
    '''
            
    
if __name__ == '__main__':
    train_qp_network()