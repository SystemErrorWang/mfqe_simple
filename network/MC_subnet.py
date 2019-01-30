#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:46:04 2019

@author: secret_wang
"""

import torch
import torch.nn as nn
from torch.nn import init
from module.Transformer import Transformer, trans_func



class MotionCompensateSubnet(nn.Module):
    
    def __init__(self):
        super(MotionCompensateSubnet, self).__init__()
        self.downsample_4x = nn.Sequential(
                nn.Conv2d(2, 24, kernel_size=5, stride=2, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=5, stride=2, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
                )
        self.ps_4x = nn.PixelShuffle(4)
        
        self.downsample_2x = nn.Sequential(
                nn.Conv2d(5, 24, kernel_size=5, stride=2, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(24),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 8, kernel_size=3, stride=1, padding=1, bias=False),
                )
        self.ps_2x = nn.PixelShuffle(2)
        
        self.pixelwise_mc = nn.Sequential(
                nn.Conv2d(5, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=5, stride=1, padding=2, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(24, 2, kernel_size=3, stride=1, padding=1, bias=False),
                )
        
        self.ps_1x =nn.PixelShuffle(1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.15, 
                                     mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.15, 
                                     mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.0256)
                init.constant_(m.bias.data, 0.0)

    '''
    
    def __init__(self):
        super(MotionCompensateSubnet, self).__init__()
        self.downsample_4x = nn.Sequential(
                nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                )
        self.ps_4x = nn.PixelShuffle(4)
        
        self.downsample_2x = nn.Sequential(
                nn.Conv2d(5, 32, kernel_size=5, stride=2, padding=2, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False),
                )
        self.ps_2x = nn.PixelShuffle(2)
        
        self.pixelwise_mc = nn.Sequential(
                nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=False),
                #nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False),
                )
        
        self.ps_1x =nn.PixelShuffle(1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.15, 
                                     mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.15, 
                                     mode='fan_in', nonlinearity='relu')
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.0256)
                init.constant_(m.bias.data, 0.0)
                
    '''
        
    def forward(self, image_a, image_b):
        
        batch_size, channel, h, w = image_a.size()
        
        cat_4x = torch.cat((image_a, image_b), 1)
        out_4x = self.downsample_4x(cat_4x)
        out_4x = self.ps_4x(torch.tanh(out_4x))
        #transformer_4x = Transformer(out_4x, image_b)
        #wrap_4x = transformer_4x()
        wrap_4x = trans_func(out_4x, image_b)
        
        cat_2x = torch.cat((cat_4x, out_4x, wrap_4x), 1)
        out_2x = self.downsample_2x(cat_2x)
        out_2x = self.ps_2x(torch.tanh(out_2x))
        add_2x = out_2x + out_4x
        #transformer_2x = Transformer(add_2x, image_b)
        #wrap_2x = transformer_2x()
        wrap_2x = trans_func(add_2x, image_b)
        
        cat_1x = torch.cat((cat_4x, add_2x, wrap_2x), 1)
        out_1x = self.pixelwise_mc(cat_1x)
        out_1x = self.ps_1x(torch.tanh(out_1x))
        add_1x = out_1x + add_2x
        #transformer_1x = Transformer(add_1x, image_b)
        #wrap_1x = transformer_1x()
        wrap_1x = trans_func(add_1x, image_b)
        
        return wrap_1x
    
    
'''
image_a = torch.ones(4, 1, 360, 540)
image_b = torch.ones(4, 1, 360, 540)
mcnet = MotionCompensateSubnet()
print(mcnet(image_a, image_b).size())
'''
        
        
        
        
        
        
        
        