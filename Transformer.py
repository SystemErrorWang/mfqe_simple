#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 14:18:42 2019

@author: secret_wang
"""

import torch
import torch.nn as nn
import tensorflow as tf

class Transformer(nn.Module):
    def __init__(self, flow, pqf, cuda=True):
        super(Transformer, self).__init__()
        self.batch = pqf.size()[0]
        self.channel = pqf.size()[1]
        self.flow = flow
        self.pqf = pqf
        self.cuda = cuda
        self.out_size = pqf.size()[2:]
        
    def repeat(self, x, n_repeats):
        rep = torch.ones(size = [n_repeats])
        rep = rep.unsqueeze(1).transpose(1, 0)
        if self.cuda:
            x = x.cuda()
            rep = rep.cuda()
        x = torch.matmul(x.view(-1, 1).float(), rep).view(-1)
        return x
    
    def repeat2(self, x, n_repeats):
        rep = torch.ones(size = [n_repeats])
        rep = rep.unsqueeze(1)
        if self.cuda:
            x = x.cuda()
            rep = rep.cuda()
        x = torch.matmul(rep, x.view(1, -1).float()).view(-1)
        return x
    
    def interpolate(self, im, x, y, out_size):
        batch_size, channels = im.size()[:2]
        height, width = im.size()[2:]
        x, y = x.to(torch.float32), y.to(torch.float32)
        #height_f, width_f = height.to(torch.float32), width.to(torch.float32)
        out_height, out_width = out_size[:2]
        zero = torch.zeros([], dtype=torch.float32)
        max_y, max_x = int(im.size()[1] - 1), int(im.size()[2] - 1)
        x = self.repeat2(torch.arange(width), 
                        height * batch_size) + x * 64
        y = self.repeat2(self.repeat(torch.arange(height), width), 
                         batch_size) + y * 64
                         
        x0 = torch.floor(x)
        x1 = x0 + 1
        y0 = torch.floor(y)
        y1 = y0 + 1
        x0, x1 = x0.clamp(zero, max_x), x1.clamp(zero, max_x)
        y0, y1 = y0.clamp(zero, max_y), y1.clamp(zero, max_y)
        dim1, dim2 = width, width * height
        base = self.repeat(torch.arange(batch_size) * dim1, out_height * out_width)
        base_y0, base_y1 = base + y0 * dim2, base + y1 * dim2
        idx_a, idx_b = base_y0 + x0, base_y1 + x0
        idx_c, idx_d = base_y0 + x1, base_y1 + x1
        
        im_flat = im.view(-1, channels).to(torch.float32)
        #i_a, i_b = im_flat.index_select(idx_a), im_flat.index_select(idx_b)
        #i_c, i_d = im_flat.index_select(idx_c), im_flat.index_select(idx_d) 
        i_a, i_b = im_flat[idx_a.long()], im_flat[idx_b.long()]
        i_c, i_d = im_flat[idx_c.long()], im_flat[idx_d.long()]
        
        x0_f, x1_f = x0.to(torch.float32), x1.to(torch.float32)
        y0_f, y1_f = y0.to(torch.float32), y1.to(torch.float32)
        wa = torch.unsqueeze((x1_f - x) * (y1_f - y), 1)
        wb = torch.unsqueeze((x1_f - x) * (y - y0_f), 1)
        wc = torch.unsqueeze((x - x0_f) * (y1_f - y), 1)
        wd = torch.unsqueeze((x - x0_f) * (y - y0_f), 1)
        output = wa*i_a + wb*i_b + wc*i_c + wd*i_d
        return output
    
    
    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones_like(torch.stack([height, 1])),
                           torch.linspace(-1.0, 1.0, width).unsqueeze(1).transpose(1, 0))
        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).unsqueeze(1),
                           torch.ones_like(torch.stack([1, width])))
        x_t_flat, y_t_flat = x_t.view(1, -1), y_t.view(1, -1)
        grid = torch.cat([x_t_flat, y_t_flat, torch.ones_like(x_t_flat)], dim = 0)
        return grid
    
    
    def transform(self, x_s, y_s, input_dim, out_size):
        num_batch, num_channels = input_dim.size()[:2]
        height, width = input_dim.size()[2:]
        #height_f, width_f = height.to(torch.float32), width.to(torch.float32)
        out_height, out_width = out_size
        x_s_flat = x_s.contiguous().view(-1)
        y_s_flat = y_s.contiguous().view(-1)
        
        input_transform = self.interpolate(input_dim, x_s_flat, 
                                           y_s_flat, out_size)
        output = input_transform.view(self.batch, self.channel, out_height, out_width)
        return output
    
    
    def __call__(self):
        dx, dy = torch.split(self.flow, 1, 1)
        output = self.transform(dx, dy, self.pqf, self.out_size)
        return output
        

'''
imgb = torch.ones(4, 1, 360, 540).cuda()
c5_hr = torch.ones(4, 2, 360, 540).cuda()
transformer = Transformer(c5_hr, imgb)
print(transformer().size())
'''

    

                