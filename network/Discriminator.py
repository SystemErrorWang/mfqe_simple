import sys
sys.path.append("..")
import torch
import torch.nn as nn
from module.spectral_norm import spectral_norm


class DiscBatchNorm(nn.Module):
    def __init__(self, in_channel, base_channel=32):
        super(DiscBatchNorm, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel, base_channel, 
                          kernel_size=3, stride=1, bias=True, padding=1),
                nn.BatchNorm2d(base_channel),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel, base_channel, 
                          kernel_size=4, stride=2, bias=True, padding=1),
                nn.BatchNorm2d(base_channel),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel, base_channel*2, 
                          kernel_size=3, stride=1, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*2),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*2, base_channel*2, 
                          kernel_size=4, stride=2, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*2),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*2, base_channel*4, 
                          kernel_size=3, stride=1, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*4),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*4, base_channel*4, 
                          kernel_size=4, stride=2, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*4),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*4, base_channel*8, 
                          kernel_size=3, stride=1, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*8),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*8, base_channel*8, 
                          kernel_size=4, stride=2, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*8),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*8, base_channel*8, 
                          kernel_size=3, stride=1, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*8),
                nn.LeakyReLU(0.1, inplace=True),
                
                nn.Conv2d(base_channel*8, base_channel*8, 
                          kernel_size=4, stride=2, bias=True, padding=1),
                nn.BatchNorm2d(base_channel*8),
                nn.LeakyReLU(0.1, inplace=True),
                )
                
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channel*8, 1)
            
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).view(x.size()[0], -1)
        x = self.classifier(x)
        return x
    
    
    
class DiscSpecNorm(nn.Module):
    def __init__(self, in_channel, base_channel=32):
        super(DiscSpecNorm, self).__init__()
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.conv0 = spectral_norm(nn.Conv2d(in_channel, base_channel, 
                                kernel_size=3, stride=1, bias=True, padding=1))
        self.conv1 = spectral_norm(nn.Conv2d(base_channel, base_channel, 
                                kernel_size=4, stride=2, bias=True, padding=1))
        
        self.conv2 = spectral_norm(nn.Conv2d(base_channel, base_channel*2, 
                                kernel_size=3, stride=1, bias=True, padding=1))
        self.conv3 = spectral_norm(nn.Conv2d(base_channel*2, base_channel*2, 
                                kernel_size=4, stride=2, bias=True, padding=1))
        
        self.conv4 = spectral_norm(nn.Conv2d(base_channel*2, base_channel*4, 
                                kernel_size=3, stride=1, bias=True, padding=1))
        self.conv5 = spectral_norm(nn.Conv2d(base_channel*4, base_channel*4, 
                                kernel_size=4, stride=2, bias=True, padding=1))
        
        self.conv6 = spectral_norm(nn.Conv2d(base_channel*4, base_channel*8, 
                                kernel_size=3, stride=1, bias=True, padding=1))
        self.conv7 = spectral_norm(nn.Conv2d(base_channel*8, base_channel*8, 
                                kernel_size=4, stride=2, bias=True, padding=1))
        
        self.conv8 = spectral_norm(nn.Conv2d(base_channel*8, base_channel*8, 
                                kernel_size=3, stride=1, bias=True, padding=1))
        self.conv9 = spectral_norm(nn.Conv2d(base_channel*8, base_channel*8, 
                                kernel_size=4, stride=2, bias=True, padding=1))
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channel*8, 1)
        
    def forward(self, x):
        x = self.act(self.conv0(x))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.act(self.conv7(x))
        x = self.act(self.conv8(x))
        x = self.act(self.conv9(x))
        x = self.pool(x).view(x.size()[0], -1)
        x = self.classifier(x)
        return x
        
'''
test = torch.ones(4, 1, 128, 128)
model = DiscSpecNorm(1, 16)
out = model(test)
print(out.size())
'''
        