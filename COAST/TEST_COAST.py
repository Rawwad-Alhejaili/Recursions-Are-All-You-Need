# -*- coding: utf-8 -*-

# import os
# cwd = os.getcwd() # Current working directory
# print(os.getcwd())  # Prints the current working directory
# username = 'ruwwadalhejaily'
# if cwd[6:21] == username:
#     remote = True
#     os.chdir('/home/ruwwadalhejaily/Codes/Python/Recursions-Are-All-You-Need/COAST')
#     print(os.getcwd())  # Prints the current working directory
#     os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  #why? Preserves GPU order?
#     os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used
#     import torch
#     print(f'Available GPU Devices: {torch.cuda.device_count()}')
# else:
#     remote = False

import os
import torch
import numpy as np
from argparse import ArgumentParser

from show_image import show_image
from COAST import COAST
from COAST import R_COAST


# =============================================================================
# Preallocation (to some extent)
# =============================================================================
parser = ArgumentParser(description='COAST')
parser.add_argument('--gpu_list',      type=str,   default='0',               help='gpu index')
parser.add_argument('--matrix_dir', type=str, default='sampling_matrix/test', help='sampling matrix directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
args = parser.parse_args()

gpu_list = args.gpu_list

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list  # specify which GPU(s) to be used
gpu_count = torch.cuda.device_count()
print(f'Available GPU Devices: {gpu_count}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  #Release GPU memory from cache

# ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}
ratio_dict = {1: 11, 4: 44, 5:54, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 544}
n_output = 1089  
nrtrain = 88912  # number of training blocks #Moved below
batch_size = 64
total_phi_num = 1
rand_num = 1

test_cs_ratio_set = [10, 20, 30, 40, 50]

Phi = {}
for cs_ratio in test_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, n_output))
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    Phi_data = np.load(Phi_name)[:rand_num]
    # for k in range(rand_num):
    Phi[cs_ratio] = Phi_data

# Convert Phi to a tensor
for cs_ratio in test_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi[cs_ratio]).type(torch.FloatTensor)
    # Phi[cs_ratio] = Phi[cs_ratio].to(device)
    
del total_phi_num, ratio_dict, Phi_data, Phi_name, size_after_compress


#%%
# =============================================================================
# Load the COAST 20 Layers No Recursion model
# =============================================================================
model_name = 'COAST'
model_path = 'model/Full_Data/COAST_layer_20_IPL_1_lr_1.00e-04_nrtrain=88912/Best_PSNR/epoch=388_PSNR_SSIM=28.66_0.8589_IPL=1_checkpoint.pkl'

feedback = [1]*20
group_num = feedback
layer_num = len(feedback)
block_size=[33,33]

model = COAST(layer_num, feedback=feedback, block_size=block_size)
model.load_state_dict(torch.load(model_path)['model'])


# =============================================================================
# Obtain its results
# =============================================================================

#%% Set11 Results
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, test_cs_ratio_set, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='Set11', model_name=model_name, 
                                    feedback=feedback)

#%% BSD68
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, CS_ratio, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='SetBSD68', model_name=model_name, 
                                    feedback=feedback)


#%%
# =============================================================================
# Load the R-COAST 5 Layers 4 iterations/layer model
# =============================================================================
model_name = 'R-COAST'
model_path = 'model/Full_Data/R-COAST_layer_5_IPL_4_lr_1.00e-04_nrtrain=88912/Best_PSNR/epoch=368_PSNR_SSIM=28.50_0.8543_IPL=4_checkpoint.pkl'

feedback = [1]*5
IPL = 4
feedback_max = [IPL]*5
group_num = feedback
layer_num = len(feedback)
block_size=[33,33]

model = R_COAST(layer_num, feedback=feedback, block_size=block_size, IPL=IPL)
model.load_state_dict(torch.load(model_path)['model'])


# =============================================================================
# Obtain its results
# =============================================================================

#%% Set11 Results
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, CS_ratio, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='Set11', model_name=model_name, 
                                    feedback=feedback_max)

#%% BSD68
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, CS_ratio, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='SetBSD68', model_name=model_name, 
                                    feedback=feedback_max)


#%%
# =============================================================================
# Limited Data
# =============================================================================
print(r"Results when using only 3% of the Training Data")

#%%
# =============================================================================
# Load the COAST 20 Layers No Recursion model
# =============================================================================
model_name = 'COAST'
model_path = 'model/Limited_Data/COAST_layer_20_IPL_1_lr_1.00e-04_nrtrain=2667/Best_PSNR/epoch=382_PSNR_SSIM=27.88_0.8405_IPL=1_checkpoint.pkl'

feedback = [1]*20
group_num = feedback
layer_num = len(feedback)
block_size=[33,33]

model = COAST(layer_num, feedback=feedback, block_size=block_size)
model.load_state_dict(torch.load(model_path)['model'])


# =============================================================================
# Obtain its results
# =============================================================================

#%% Set11 Results
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, test_cs_ratio_set, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='Set11', model_name=model_name, 
                                    feedback=feedback)
#%% BSD68
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, CS_ratio, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='SetBSD68', model_name=model_name, 
                                    feedback=feedback)

#%%
# =============================================================================
# Load the R-COAST 5 Layers 4 iterations/layer model
# =============================================================================
model_name = 'R-COAST'
model_path = 'model/Limited_Data/R-COAST_layer_5_IPL_4_lr_1.00e-04_nrtrain=2667/Best_PSNR/epoch=349_PSNR_SSIM=27.98_0.8430_IPL=4_checkpoint.pkl'

feedback = [1]*5
IPL = 4
feedback_max = [IPL]*5
group_num = feedback
layer_num = len(feedback)
block_size=[33,33]

model = R_COAST(layer_num, feedback=feedback, block_size=block_size, IPL=IPL)
model.load_state_dict(torch.load(model_path)['model'])


# =============================================================================
# Obtain its results
# =============================================================================

#%% Set11 Results
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, CS_ratio, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='Set11', model_name=model_name, 
                                    feedback=feedback_max)

#%% BSD68
test_cs_ratio_set = [10, 20, 30, 40, 50]
CS_ratio = test_cs_ratio_set
set11_psnr, set11_ssim = show_image(model, Phi, CS_ratio, args, 
                                    batch_size=1024*1, img_no='all', 
                                    test_name='SetBSD68', model_name=model_name, 
                                    feedback=feedback_max)