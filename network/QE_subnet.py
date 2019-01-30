#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:46:58 2019

@author: secret_wang
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import init

class QualityEnhanceSubnet(nn.Module):
    
    def __init__(self):
        super(QualityEnhanceSubnet, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False),
                nn.PReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False),
                nn.PReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False),
                nn.PReLU()
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.PReLU()
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.PReLU()
                )
        self.conv6 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PReLU()
                )
        self.conv7 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.PReLU()
                )
        self.conv8 = nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.PReLU()
                )
        self.conv9 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
                init.constant_(m.bias.data, 0.0)
           
        
        
    def forward(self, x1, x2, x3):
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)
        conv3 = self.conv3(x3)
        concat12 = torch.cat((conv1, conv2), 1)
        concat23 = torch.cat((conv2, conv3), 1)
        conv4 = self.conv4(concat12)
        conv5 = self.conv5(concat23)
        conv6 = self.conv6(conv4)
        conv7 = self.conv7(conv5)
        concat67 = torch.cat((conv6, conv7), 1)
        conv8 = self.conv8(concat67)
        conv9 = self.conv9(conv8)
        output = x2 + conv9
        return output
        
    

class BNQENet(nn.Module):

    def __init__(self):
        super(QualityEnhanceSubnet, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False),
                nn.BatchNorm2d(128),
                nn.PReLU()
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False),
                nn.BatchNorm2d(128),
                nn.PReLU()
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=9, stride=1, padding=4, bias=False),
                nn.BatchNorm2d(128),
                nn.PReLU()
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU()
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=7, stride=1, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU()
                )
        self.conv6 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU()
                )
        self.conv7 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.PReLU()
                )
        self.conv8 = nn.Sequential(
                nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(32),
                nn.PReLU()
                )
        self.conv9 = nn.Conv2d(32, 1, kernel_size=5, stride=1, padding=2, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0)
                init.constant_(m.bias.data, 0.0)
        
        
    def forward(self, x1, x2, x3):
        conv1 = self.conv1(x1)
        conv2 = self.conv2(x2)
        conv3 = self.conv3(x3)
        concat12 = torch.cat((conv1, conv2), 1)
        concat23 = torch.cat((conv2, conv3), 1)
        conv4 = self.conv4(concat12)
        conv5 = self.conv5(concat23)
        conv6 = self.conv6(conv4)
        conv7 = self.conv7(conv5)
        concat67 = torch.cat((conv6, conv7), 1)
        conv8 = self.conv8(concat67)
        conv9 = self.conv9(conv8)
        output = x2 + conv9
        return output

'''
for i in tqdm(range(100)):
    a = torch.ones(4, 1, 36, 54)
    b = torch.ones(4, 1, 36, 54)
    c = torch.ones(4, 1, 36, 54)
    qenet = QualityEnhanceSubnet()
    out = qenet(a, b, c)
'''

        