from collections import OrderedDict
import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel, growth_channel=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(in_channel, growth_channel, 
                                             kernel_size=3, stride=1, bias=bias, padding=1))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(in_channel+growth_channel, growth_channel, 
                                             kernel_size=3, stride=1, bias=bias, padding=1))
        self.conv3 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(in_channel+2*growth_channel, growth_channel, 
                                             kernel_size=3, stride=1, bias=bias, padding=1))
        self.conv4 = nn.Sequential(nn.LeakyReLU(0.1, inplace=True),
                                   nn.Conv2d(in_channel+3*growth_channel, growth_channel, 
                                             kernel_size=3, stride=1, bias=bias, padding=1))
        self.conv5 = nn.Conv2d(in_channel+4*growth_channel, growth_channel, 
                               kernel_size=3, stride=1, bias=bias, padding=1)
                
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(0.2) + x



class RRDBNet(nn.Module):
    def __init__(self, in_channel, out_channel, mid_channel=32, growth_channel=32, num_blocks=16):
        super(RRDBNet, self).__init__()
        
        self.fea_conv = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1)
        
        self.bulid_block(mid_channel, growth_channel, num_blocks)
                                                  
        self.lr_conv = nn.Conv2d(mid_channel, mid_channel, kernel_size=3, stride=1, padding=1)

        self.hr_conv = nn.Sequential(nn.Conv2d(mid_channel, mid_channel, 
                                                kernel_size=3, stride=1, padding=1),
                                        nn.LeakyReLU(0.1, inplace=False)
                                        )
        self.final_conv = nn.Conv2d(mid_channel, out_channel, 
                                    kernel_size=3, stride=1, padding=1)
        
    def bulid_block(self, in_channel, growth_channel, num_blocks):
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualDenseBlock(in_channel, growth_channel))
        self.rb_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.fea_conv(x)
        x = x + self.rb_blocks(x)
        x = self.hr_conv(x)
        x = self.final_conv(x)
        return x
    
    
'''
model = RRDBNet(1, 1, 16, 16, 8)
test = torch.ones(4, 1, 32, 32)
out = model(test)
print(out.size())
'''