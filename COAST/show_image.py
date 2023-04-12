#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy
import math
import os
from argparse import ArgumentParser
import glob
import torch
from time import time
from tqdm import tqdm
import cv2
from prettytable import PrettyTable
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim
    
from block import Block

def show_image(model_tr, Phi, cs_ratios, args, batch_size=1024, block_size=None, 
               feedback=None, img_no=None, test_name='MY_TEST_DATASET',
               model_name='COAST_Feedback', disp_image=False):
    
    if model_name.lower() == 'r-coast' or model_name.lower() == 'r_coast':
        from COAST import R_COAST as COAST_Feedback
    elif model_name.lower() == 'coast':
        from COAST import COAST as COAST_Feedback
        
    import matplotlib.pyplot as plt
    from lazy_imshow import lazy_imshow
    lazy_show = lazy_imshow(I              = None, 
                            title          = None, 
                            grid           = (1,2), 
                            colorbar       = False, 
                            cbar_ticks     = 11,
                            aspect         = 1,
                            cmap           = 'gray', 
                            pad            = 0.7, 
                            plotsize       = (7,7), 
                            rang           = 'norm', 
                            rangZeroCenter = False,
                            dpi            = 300, 
                            figTransparent = False, 
                            fontsize       = 14, 
                            alphabet       = False,
                            ignoreZeroStd  = False, 
                            clip           = 1,
                            t              = None,
                            offset         = None,
                            fft            = False)
    imshow = lazy_show.imshow
    
    table = PrettyTable(['Model'] + [str(i) for i in cs_ratios] + ['Run Time'])
    
    if type(cs_ratios) == type(0):
        cs_ratios = [cs_ratios]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if feedback is None:
        try:
            feedback = model_tr.feedback
        except:
            feedback = model_tr.module.feedback
        
    if block_size is None:
        try:
            block_size = model_tr.block_size
        except:
            block_size = model_tr.module.block_size
    
    try:
        max_recursion = model_tr.max_recursion
        model = COAST_Feedback(LayerNo=len(feedback), feedback=feedback, max_recursion=max_recursion, block_size=block_size)
    except:
        model = COAST_Feedback(LayerNo=len(feedback), feedback=feedback, block_size=block_size)
        
    try:
        model.to(device).load_state_dict(model_tr.state_dict())
    except:
        model = torch.nn.DataParallel(model)
        model.to(device).load_state_dict(model_tr.state_dict())
    
    def psnr(img1, img2):
        img1.astype(np.float32)
        img2.astype(np.float32)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    
    
    test_dir = os.path.join(args.data_dir, test_name)
    filepaths = glob.glob(test_dir + '/*')
    ImgNum = len(filepaths)
        
        
            
    if type(img_no) == type([]):
        iterations = img_no
    elif img_no is None:
        iterations = [np.random.randint(ImgNum)]
    elif str(img_no).lower() == 'all':
        iterations = range(ImgNum)
    elif img_no > ImgNum-1 or img_no < -ImgNum:
        print('img_no is greater than the number of images in the dataset.')
        print('Setting img_no to a random image')
        iterations = [np.random.randint(ImgNum)]
    elif type(img_no) == type(0):
        img_no = [img_no]
        iterations = img_no
    else:
        return print('I dunno how to fix this anymore ):')
    
    PSNR_All = np.zeros([len(cs_ratios), len(iterations)], dtype=np.float32)
    SSIM_All = np.zeros([len(cs_ratios), len(iterations)], dtype=np.float32)
    
    print('\n')
    print("CS Reconstruction Start")
    start = time()
    
    with torch.no_grad():
        for k, csr in enumerate(cs_ratios):
            for i, img_no in enumerate(iterations):
                imgName = filepaths[img_no]
        
                Img = cv2.imread(imgName, 1)
                Img_yuv = cv2.cvtColor(Img, cv2.COLOR_BGR2YCrCb)
                Img_rec_yuv = Img_yuv.copy()
                Img = Img[:,:,:3]
        
                Iorg_y = Img_yuv[:,:,0]
                Iorg = Iorg_y
                Iorg_y = torch.tensor(Iorg_y)
                h,w = Iorg_y.shape
                Iorg_y = Iorg_y.reshape(1,1,h,w)
                cur_Phi = Phi[csr][0].to(device)
                
                # start = time()
                Irec = model(Iorg_y.to(device)/255.0, cur_Phi, block_size=block_size)[-1][-1].cpu().data.numpy().squeeze()
                        
                # end = time()
                
                X_rec = np.clip(Irec, 0, 1)
        
                rec_PSNR = psnr(X_rec*255, Iorg.astype(np.float64))
                rec_SSIM = ssim(X_rec*255, Iorg.astype(np.float64), data_range=255)
        
                # print("[%02d/%02d] For %s, PSNR is %.2f, SSIM is %.4f" % (img_no+1, ImgNum, imgName, rec_PSNR, rec_SSIM))
        
                Img_rec_yuv[:,:,0] = X_rec*255
        
                im_rec_rgb = cv2.cvtColor(Img_rec_yuv, cv2.COLOR_YCrCb2BGR)
                im_rec_rgb = np.clip(im_rec_rgb, 0, 255).astype(np.uint8)
                
                Img = np.array([Img[:,:,2], Img[:,:,1], Img[:,:,0]])
                im_rec_rgb = np.array([im_rec_rgb[:,:,2], im_rec_rgb[:,:,1], im_rec_rgb[:,:,0]])
                h,w = Img.shape[1:]
                if disp_image:
                    titles = ['Ground Truth\n', f"CS_Ratio={csr}% \nPSNR={rec_PSNR:.2f} SSIM={rec_SSIM:.4f}"]
                    # resultName = imgName.replace(args.data_dir, args.result_dir)
                    # titles = [f"{resultName}\n"[23:], f"{resultName} CS_Ratio={cs_ratio}% \nPSNR={rec_PSNR:.2f} SSIM={rec_SSIM:.4f}"[23:]]
                    if h < w:
                        imshow([Iorg_y, X_rec*255],       title=titles, grid=(2,1))
                        if test_name.lower() != 'set11':
                            imshow([Img/255, im_rec_rgb/255], title=titles, grid=(2,1))
                    else:
                        imshow([Iorg_y, X_rec*255],       title=titles, grid=(1,2))
                        if test_name.lower() != 'set11':
                            imshow([Img/255, im_rec_rgb/255], title=titles, grid=(1,2))
                plt.show()
                # cv2.imwrite("%s_ISTA_Net_ratio_%d_epoch_%d_PSNR_%.2f_SSIM_%.4f.png" % (resultName, cs_ratio, epoch_num, rec_PSNR, rec_SSIM), im_rec_rgb)
                del Irec
        
                PSNR_All[k, i] = rec_PSNR
                SSIM_All[k, i] = rec_SSIM
            
            end = time()
            
            run_time = end-start
        
            print('\n')
            output_data = f"Run time for {test_name} is {run_time:.3f}"
            print(output_data)
            
            # print('\n')
            output_data = "CS ratio is %d, Avg PSNR/SSIM for %s is %.2f/%.4f \n" % (csr, test_name, PSNR_All[k].mean(), SSIM_All[k].mean())
            print(output_data)
        
        table.add_row([model_name] + [f'{PSNR_All[j].mean():.2f}/{SSIM_All[j].mean():.4f}' for j in range(len(cs_ratios))] + [f'{run_time:.3f}'])
        
    print(table)
    
    return PSNR_All, SSIM_All
