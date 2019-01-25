#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:30:12 2019

@author: secret_wang
"""

import torch 
import torch.nn as nn
from torch.nn import init
from tqdm import tqdm


class QPBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(QPBlock, self).__init__()
        self.main_branch = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                                  groups=8, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(out_channel, out_channel, kernel_size=1, 
                                  stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(out_channel)
                        )
        self.shortcut = nn.Conv2d(in_channel, out_channel, kernel_size=1, 
                                  stride=1, padding=0, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    
        
    def forward(self, x):
        shortcut = self.pool(self.shortcut(x))
        main_branch = self.main_branch(x)
        output = self.act(shortcut+main_branch)
        return output
    


class QualityPredictNetwork(nn.Module):
    
    '''
    this is the network used to evaluate and predict 
    the quality of a certain image.
    input size should be (batch_size, 1, hight, width)
    '''
    
    def __init__(self):
        super(QualityPredictNetwork, self).__init__()
        
        #self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(0.1, inplace=True)
                        )
        
        self.block1 = QPBlock(32, 48)
        
        self.block2 = QPBlock(48, 64)
        
        self.block3 = QPBlock(64, 96)
        
        self.block4 = QPBlock(96, 128)
        
        self.block5 = QPBlock(128, 192)
        
        self.block6 = QPBlock(192, 256)
        
        self.pool_final = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(256, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data, a=0.15, 
                                     mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, a=0.15, 
                                     mode='fan_in', nonlinearity='leaky_relu')
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.0256)
                init.constant_(m.bias.data, 0.0)
                
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool_final(out)
        out = self.fc(out.view(out.size()[0], -1))
        return out
        

'''
network = QualityEvaluationNetwork()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters, lr=1e-3, nesterov=True)

for i in tqdm(range(100)):
    a = torch.ones(4, 1, 360, 540)
    qpnet = QualityPredictNetwork()
print(qpnet(a).size())
'''

        
        