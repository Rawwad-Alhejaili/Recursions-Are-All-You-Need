#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from copy import deepcopy

from block import Block

#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CPMB(nn.Module):
    '''Residual block with scale control
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    
    def __init__(self, res_scale_linear, nf=32):
        super().__init__()
        
        self.nf = nf
        conv_bias = True
        scale_bias = True
        map_dim = 64
        cond_dim = 2

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)

        self.res_scale = res_scale_linear

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]
        # cond = cond[:, :1]

        cond_repeat = cond.repeat((content.shape[0], 1))

        out = self.act(self.conv1(content))
        out = self.conv2(out)

        res_scale = self.res_scale(cond_repeat)
        alpha1 = res_scale.view(-1, self.nf, 1, 1)

        out1 = out * alpha1
        return content + out1, cond


class BasicBlock(torch.nn.Module):
    def __init__(self, res_scale_linear):
        super().__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.head_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32)
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiTPhi, PhiTb, cond, B):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        
        x = B.unvectorize(x)
        x = B.unblockify(x)

        x_mid = self.head_conv(x)
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)

        x_pred = x + x_mid
        
        
        x_pred = B.blockify(x_pred)
        x_pred = B.vectorize(x_pred)

        return x_pred
    
    
class COAST(torch.nn.Module):
    def __init__(self, LayerNo, feedback, block_size=(33,33)):
        super().__init__()
        
        onelayer = []
        self.LayerNo = LayerNo
        self.block_size = block_size
        nf = 32
        scale_bias = True
        res_scale_linear = nn.Linear(1, nf, bias=scale_bias)

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear))

        self.fcs = nn.ModuleList(onelayer)
        self.feedback = feedback
        assert len(feedback) == LayerNo
        
        
    def get_cond(self, cs_ratio, cond_type):
        # para_noise = sigma / 5.0
        if cond_type == 'org_ratio':
            para_cs = cs_ratio / 100.0
        else:
            para_cs = cs_ratio * 2.0 / 100.0
        
        para = torch.tensor([[para_cs]])
    
        return para

    def forward(self, x_input, Phi, block_size=None, feedback=None):
        
        if feedback is None:
            feedback = self.feedback
        else:
            assert len(feedback) == self.LayerNo
            
        if block_size is None:
            block_size = self.block_size
        
        x = x_input.clone()
        h,w = x.shape[-2:]
        
        B = Block(block_size)
        Iblk = B.blockify(x)
        Icol = B.vectorize(Iblk)
        
        batch_x = Icol
        cs_ratio = (Phi.shape[0] / Phi.shape[1])*100
        cond = self.get_cond(cs_ratio, 'org_ratio').to(device)
        
        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))  #100% correct

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = PhiTb.clone()
        
        outputs = [[] for i in range(self.LayerNo)] 
        for i, recursions in enumerate(feedback):
            for j in range(recursions):
                x = self.fcs[i](x, PhiTPhi, PhiTb, cond, B)
                out = B.unvectorize(x)
                out = B.unblockify(out)
                outputs[i].append(out)
                
        try:
            outputs[-1][-1] = out
        except:
            outputs[-1].append(out)

        return outputs


class R_COAST(torch.nn.Module):
    def __init__(self, LayerNo, feedback, max_recursion, block_size=(33,33)):
        super().__init__()
        
        self.feedback = feedback
        feedback = [1] * LayerNo  #temp
        onelayer = []
        self.LayerNo = LayerNo
        self.max_recursion = max_recursion
        self.block_size = block_size
        nf = 32
        scale_bias = True
        res_scale_linear = nn.Sequential(
        nn.Linear(3, nf, bias=scale_bias),
        nn.ReLU(),
        nn.Linear(nf, nf, bias=scale_bias)
        )

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear))

        self.fcs = nn.ModuleList(onelayer)
        
        feedback_max = [feedback[0]*max_recursion[0]] \
                        + (np.array(feedback[1:-1])*max_recursion[1]).tolist() \
                        + [feedback[-1]*max_recursion[2]]
                        
        self.max_iter = sum(feedback_max)
        self.max_recursion = max_recursion
        assert len(feedback) == LayerNo
        
        
    def get_cond(self, cs_ratio, total_recursions, cur_recursion):
        para_total_recursions = total_recursions / (self.max_iter)
        para_cur_recursion = cur_recursion / (self.max_iter)
        para_cs = cs_ratio / 100.0 * 2  #so that max(para_cs) == 1
        
        para = torch.tensor([[para_cs, 
                              para_total_recursions, 
                              para_cur_recursion]])
        return para

    def forward(self, x_input, Phi, block_size=None, feedback=None):
        
        if feedback is None:
            feedback = self.feedback
        else:
            assert len(feedback) == self.LayerNo
            
        if block_size is None:
            block_size = self.block_size
        
        x = x_input.clone()
        
        B = Block(block_size)
        Iblk = B.blockify(x)
        Icol = B.vectorize(Iblk)
        
        batch_x = Icol
        cs_ratio = (Phi.shape[0] / Phi.shape[1])*100
        
        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = PhiTb.clone()
        
        outputs = [[] for i in range(self.LayerNo)] 
        iteration = 0
        total_iter = sum(feedback)
        for i, recursions in enumerate(feedback):
            for j in range(recursions):
                iteration += 1
                cond = self.get_cond(cs_ratio,total_iter,iteration).to(device)
                x = self.fcs[i](x, PhiTPhi, PhiTb, cond, B)
                out = B.unvectorize(x)
                out = B.unblockify(out)
                outputs[i].append(out)
        
        
        try:
            outputs[-1][-1] = out
        except:
            outputs[-1].append(out)
        
        return outputs
