#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from einops import rearrange
        
class Block():
    def __init__(self, block_dim, extraDim=False, include_C=False, shape=None):
        self.extraDim = extraDim
        self.include_C = include_C
        self.block_dim = block_dim
        self.block_h = block_dim[0]
        self.block_w = block_dim[1]
        
        if shape is not None:
            self.org_shape = shape
            self.N,self.C,self.H,self.W = shape
            self.patchX = int(np.ceil(self.H / block_dim[0]))
            self.patchY = int(np.ceil(self.W / block_dim[1]))
            self.patches_shape = (self.N, self.C, self.patchX, self.patchY, block_dim[0], block_dim[1])
        
        
    def blockify(self, I):
        
        device = I.device
        I = I.to('cpu')  #Could possibly save a little bit of VRAM this way
        self.org_shape = I.shape
        self.N,self.C,self.H,self.W = I.shape
        N,C,H,W = I.shape
        patch_row, patch_col = self.block_dim
        block_h = self.block_h
        block_w = self.block_w
        
        
        # number of patches in the x and y axes, where x=row and y=column 
        # (not intuitive, I know, but it follows textbooks' standards)
        patchX = int(np.ceil(H / patch_row))
        patchY = int(np.ceil(W / patch_col))
        self.patchX = patchX
        self.patchY = patchY
        
        
        # Pad parameters
        hr = np.mod(H, block_h)  #height reminder
        row_pad = hr and block_h - np.mod(H, block_h) #magic:) (check when hr!=0 and when hr==0)
        wr = np.mod(W, block_w)  #width reminder
        col_pad = wr and block_w - np.mod(W, block_w) #magic:) (check when wr!=0 and when wr==0)
        
        # Pad image
        Ipad = torch.nn.functional.pad(I.mT, (0,row_pad), value=0).mT
        Ipad = torch.nn.functional.pad(Ipad, (0,col_pad), value=0)
        
        
        # Blockify
        patches = Ipad.contiguous().view(N,C,patchX,patch_row,patchY,patch_col)
        patches = patches.permute(0,1,2,4,3,5).contiguous()
        
        
        self.patches_shape = patches.shape  #used to unblockify blocks later
        self.n_patches = patches.shape[1]
        
        patches = rearrange(patches, 'n c patchX patchY patch_row patch_col-> (n patchX patchY) c patch_row patch_col')
        # patches = rearrange(patches, 'n c patchX patchY patch_row patch_col-> (n patchX patchY) c patch_col patch_row')
        
        
        return patches.to(device)
    
    
    def unblockify(self, I):
        
        device = I.device
        I = I.to('cpu')
        I = rearrange(I, '(n patchX patchY) c patch_row patch_col -> n c patchX patchY patch_row patch_col',
                      patchX=self.patchX, patchY=self.patchY)
        # I = I.reshape(self.patches_shape)
        N,C,H,W = self.org_shape
        
        
        N, C, patchX, patchY, patch_row, patch_col = self.patches_shape
        
        
        output = I.contiguous().view(N,C,patchX,patchY,patch_row,patch_col)
        output = output.permute(0,1,2,4,3,5).contiguous()
        output = output.view(N,C,int(patchX * patch_row), 
                              int(patchY * patch_col)).contiguous()
        output = output[:,:,:H,:W]
        

        return output.to(device)
    
    
    def vectorize(self, I):
        # vec = rearrange(I, 'n c patchX patchY patch_row patch_col-> (n c patchX patchY) (patch_col patch_row)')
        if self.extraDim:
            vec = rearrange(I, '(n patchX patchY) c patch_row patch_col-> n (c patchX patchY) (patch_row patch_col)',
                        patchX=self.patchX, patchY=self.patchY)
        elif self.include_C:
            vec = rearrange(I, '(n patchX patchY) c patch_row patch_col-> (n patchX patchY) (c patch_row patch_col)',
                        patchX=self.patchX, patchY=self.patchY)
        else:    
            vec = rearrange(I, '(n patchX patchY) c patch_row patch_col -> (n c patchX patchY) (patch_row patch_col)',
                        patchX=self.patchX, patchY=self.patchY)
        return vec
    
    def unvectorize(self, I):
        
        # n,c = self.org_shape[:2]
        # patchX, patchY, patch_row, patch_col = self.patchX, self.patchY, self.patch_row, self.patch_col
        N, C, patchX, patchY, patch_row, patch_col = self.patches_shape
        
        if self.extraDim:
            blk = rearrange(I, 'n (c patchX patchY) (patch_row patch_col) -> (n patchX patchY) c patch_row patch_col',
                            n=N,
                            c=C, 
                            patchX=patchX, 
                            patchY=patchY, 
                            patch_row=patch_row, 
                            patch_col=patch_col)
        elif self.include_C:
            blk = rearrange(I, '(n patchX patchY) (c patch_row patch_col) -> (n patchX patchY) c patch_row patch_col',
                            n=N,
                            c=C, 
                            patchX=patchX, 
                            patchY=patchY, 
                            patch_row=patch_row, 
                            patch_col=patch_col)
        else:
            blk = rearrange(I, '(n c patchX patchY) (patch_row patch_col) -> (n patchX patchY) c patch_row patch_col',
                            n=N,
                            c=C, 
                            patchX=patchX, 
                            patchY=patchY, 
                            patch_row=patch_row, 
                            patch_col=patch_col)
            
        return blk
    
    
    
# =============================================================================
# Example
# =============================================================================
# B = Block([3,2])
# N,C,H,W = 2,1,5,3
# Im = torch.arange(N*C*H*W).reshape([N,C,H,W]).cuda()
# # t = torch.arange(1*1*4*3).reshape([1,1,4,3]).cuda()
# # I, patch_row, patch_col = [t, 2, 2]

# Im2block = B.blockify(Im)
# blk2vec = B.vectorize(Im2block)
# blk2 = B.unvectorize(blk2vec)
# blk2Im = B.unblockify(blk2)

# print('Im.shape =', Im.shape)
# print('Im2block.shape =', Im2block.shape)
# print('blk2vec.shape =', blk2vec.shape)
# print('blk2Im.shape =', blk2Im.shape)

# # Below we will run two assertions to make sure that the function is working as 
# # intended. Interestingly, for gaussian unpatchification, it cannot pass the
# # second assertion despite passing the first assertion. This is the case 
# # because even though the loss is very minimal, it is not an exact zero and as 
# # such, it cannot pass the the second assertion

# equality = ((Im-blk2Im)**2).sum().item() == 0
# print(f'Im == blk2Im is {equality}')

# assert equality



# =============================================================================
# CS Example
# =============================================================================
# from dropTracePhi import dropTracePhi
# B = Block([4,4])
# # N,C,H,W = 2,1,5,3
# N,C,H,W = 1,1,8,8
# dropRate = 0.5
# Im = 1 + torch.arange(N*C*H*W).reshape([N,C,H,W]).cuda().float()

# Im2block = B.blockify(Im)
# n,c,h,w = Im2block.shape
# sampling, drop = dropTracePhi(Im2block, dropRate, True)
# blk2vec = B.vectorize(Im2block)

# # b = blk2vec @ drop.squeeze().mT
# b = torch.mm(blk2vec, torch.transpose(drop.squeeze(), 0, 1))

# blk2 = B.unvectorize(b)
# blk2Im = B.unblockify(blk2)

# n_mis_trace = w - int(w*dropRate)
# # d = sampling.reshape(1,1,*sampling.shape)
# d = sampling
# # e = Im.mT.flatten(2).reshape(n,c,h*w,1)
# # f = d @ e
# f = torch.mm(blk2vec, torch.transpose(d, 0, 1))
# corrpt2 = f.reshape(n,c,n_mis_trace,h).mT #corrupted image

# print('Original image: \n{}\n'.format(Im))
# print('Corrupted image: \n{}\n'.format(blk2Im))
# print('Measurements: \n{}\n'.format(corrpt2))
