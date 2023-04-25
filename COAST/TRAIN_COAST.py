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
import copy
from datetime import datetime
import torch
import platform
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm
from time import time
import torch.nn as nn
import scipy.io as sio
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils import RandomDataset, write_data
from timer import timer, estimate
from generate_Gaussian_matrix import generate_Gaussian_matrix
from show_image import show_image

scaler = torch.cuda.amp.GradScaler()

# Fix the random seeds: 
seed = 1970  # "iPhones hate him" :)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
np.random.seed(seed=seed)

parser = ArgumentParser(description='COAST')

parser.add_argument('--start_epoch',   type=int,   default=0,                 help='epoch number of start training')
parser.add_argument('--end_epoch',     type=int,   default=400,               help='epoch number of end training')

# parser.add_argument('--model_name',    type=str,   default='COAST',           help='COAST or ISTA-Net+')
parser.add_argument('--RFMU',          type=bool,  default=True,             help='adds the RFMU unit')
parser.add_argument('--layer_num',     type=int,   default=5,                help='number of recovery blocks')
parser.add_argument('--IPL',           type=int,   default=4,                 help='iterations per layer (recovery block)')

parser.add_argument('--learning_rate', type=float, default=1e-4,              help='learning rate')
parser.add_argument('--gpu_list',      type=str,   default='0',               help='gpu index')
parser.add_argument('--num_workers',   type=int,   default=10,               help='number of workers')
# parser.add_argument('--group_num',     type=int,   default=1,                 help='group number for training')

parser.add_argument('--matrix_dir',    type=str,   default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir',     type=str,   default='model',           help='trained or pre-trained model directory')
parser.add_argument('--data_dir',      type=str,   default='data',            help='training data directory')
# parser.add_argument('--log_dir',       type=str,   default='log',             help='log directory')

parser.add_argument('--validation_name',     type=str,   default='Set11', help='name of validation set')
parser.add_argument('--save_cycle',    type=int,   default=10,                help='Save cycle period')

args = parser.parse_args()

start_epoch     = args.start_epoch
end_epoch       = args.end_epoch

# model_name      = args.model_name
RFMU            = args.RFMU
layer_num       = args.layer_num
IPL             = args.IPL

learning_rate   = args.learning_rate
gpu_list        = args.gpu_list
num_workers     = args.num_workers

validation_name = args.validation_name
save_cycle      = args.save_cycle

if RFMU:
    model_name = 'R-COAST'
else:
    model_name = 'COAST'

try:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True
except:
    pass

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list  # specify which GPU(s) to be used
gpu_count = torch.cuda.device_count()
print(f'Available GPU Devices: {gpu_count}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()  #Release GPU memory from cache

#%%
ratio_dict = {1: 11, 4: 44, 5:54, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 544}
n_output = 1089  
nrtrain = 88912  # number of training blocks 
# nrtrain = int(88912*0.03)  # number of training blocks  (limited data)
nrtrain_prt = copy.deepcopy(nrtrain)
batch_size = 64
total_phi_num = 50
rand_num = 50  # 1 <= rand_num <= total_phi_num

train_cs_ratio_set = [10, 20, 30, 40, 50]
# train_cs_ratio_set = np.array([*ratio_dict.keys()])

Phi = {}
for cs_ratio in train_cs_ratio_set:
    size_after_compress = ratio_dict[cs_ratio]
    Phi[cs_ratio] = np.zeros((int(rand_num * 1), size_after_compress, n_output))
    Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, n_output)
    try:
        Phi_data = np.load(Phi_name)
    except:
        Phi_data = generate_Gaussian_matrix(N=n_output, cs_ratio_set=[cs_ratio],total_phi_num=total_phi_num)
        Phi_data = Phi_data[cs_ratio]
        np.save(Phi_name, Phi_data)
    for k in range(rand_num):
        Phi[cs_ratio][k, :, :] = Phi_data[k, :, :]

# Convert Phi to a tensor
for cs_ratio in train_cs_ratio_set:
    Phi[cs_ratio] = torch.from_numpy(Phi[cs_ratio]).type(torch.FloatTensor)
    # Phi[cs_ratio] = Phi[cs_ratio].to(device)  # do this on-demand to free VRAM

# delete unnecessary variables
del total_phi_num, ratio_dict, k, Phi_data, Phi_name, size_after_compress
    
    
#%% Load the training dataset
# =============================================================================
# Load the training dataset
# =============================================================================
ratio = np.round(88912/nrtrain).astype(int)
Training_data_Name = 'Training_Data.mat'
Training_data = sio.loadmat('./%s/%s' % (args.data_dir, Training_data_Name))
Training_labels = Training_data['labels']
Training_labels = Training_labels[:nrtrain]
# idx = np.linspace(0, Training_labels.shape[0]-1, nrtrain, dtype=int)
# Training_labels = Training_labels[idx]
Training_labels = Training_labels.repeat(ratio,0)

# from block import Block
# B = Block([33,33])
# dummy = torch.ones([nrtrain,1,33,33])
# dummy = B.blockify(dummy)
# trblk = B.unvectorize(torch.tensor(Training_labels))
# a = [trblk[i:i+1] for i in range(nrtrain)]
# imshow(a, grid=[5,10], pad=0, plotsize=[1,1], dpi=300)
# del dummy, trblk, a

del Training_data_Name, Training_data

#%% Define the models
class CPMB(nn.Module):
    '''Residual block with scale control
    ---Conv-ReLU-Conv-+-
     |________________|
    '''
    
    def __init__(self, res_scale_linear, nf=32):
        super().__init__()
        
        self.nf = nf
        conv_bias = True

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)

        self.res_scale = res_scale_linear

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        content = x[0]
        cond = x[1]

        cond_repeat = cond.repeat((content.shape[0], 1))

        out = self.act(self.conv1(content))
        out = self.conv2(out)

        res_scale = self.res_scale(cond_repeat)
        alpha1 = res_scale.view(-1, self.nf, 1, 1)

        out1 = out * alpha1
        return content + out1, cond


class BasicBlock(torch.nn.Module):
    def __init__(self, res_scale_linear, nf=32):
        super().__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.head_conv = nn.Conv2d(1, nf, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            CPMB(res_scale_linear=res_scale_linear, nf=nf),
            CPMB(res_scale_linear=res_scale_linear, nf=nf),
            CPMB(res_scale_linear=res_scale_linear, nf=nf)
        )
        self.tail_conv = nn.Conv2d(nf, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiTPhi, PhiTb, cond, block_size):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, block_size, block_size)

        x_mid = self.head_conv(x_input)
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)
        
        x_pred = x_input + x_mid

        x_pred = x_pred.view(-1, block_size * block_size)

        return x_pred
    
    
class COAST(torch.nn.Module):
    def __init__(self, LayerNo, feedback, block_size=None, nf=32):
        super().__init__()
        
        onelayer = []
        self.LayerNo = LayerNo
        scale_bias = True
        res_scale_linear = nn.Linear(1, nf, bias=scale_bias)

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear,nf=nf))

        self.fcs = nn.ModuleList(onelayer)
        self.block_size = block_size
        self.feedback = feedback
        assert len(feedback) == LayerNo
            
            
    def get_cond(self, cs_ratio, cond_type):
        # para_noise = sigma / 5.0
        if cond_type == 'org_ratio':
            para_cs = cs_ratio / 100.0
        else:
            para_cs = cs_ratio * 2.0 / 100.0
        
        # para = torch.tensor([[para_cs, para_noise]])
        para = torch.tensor([[para_cs]])
    
        return para
    

    def forward(self, x, Phi, block_size=33, feedback=None):
        
        if feedback is None:
            feedback = self.feedback
        else:
            assert len(feedback) == self.LayerNo

        batch_x = x
        
        cs_ratio = (Phi.shape[0] / Phi.shape[1])*100
        cond = self.get_cond(cs_ratio, 'org_ratio').to(device)
        
        Phix = torch.mm(batch_x, torch.transpose(Phi, 0, 1))

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = PhiTb.clone()
        
        outputs = [[] for i in range(self.LayerNo)]
        for i, recursions in enumerate(feedback):
            for j in range(recursions):
                x = self.fcs[i](x, PhiTPhi, PhiTb, cond, block_size)
                outputs[i].append(x)
        outputs[-1][-1] = x

        return outputs
    

class R_COAST(torch.nn.Module):
    def __init__(self, LayerNo, feedback, IPL, block_size=None, nf=32):
        super().__init__()
        
        onelayer = []
        self.LayerNo = LayerNo
        self.IPL = IPL
        scale_bias = True
        res_scale_linear = nn.Sequential(
        nn.Linear(3, nf, bias=scale_bias),
        nn.ReLU(),
        nn.Linear(nf, nf, bias=scale_bias)
        )

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear,nf=nf))
        self.fcs = nn.ModuleList(onelayer)
        
        feedback_max = [IPL] * LayerNo
        self.max_iter = sum(feedback_max)
        self.block_size = block_size
        self.feedback = feedback
        assert len(feedback) == LayerNo
            
            
    def get_cond(self, cs_ratio, total_recursions, cur_recursion):
        para_total_recursions = total_recursions / (self.max_iter)
        para_cur_recursion = cur_recursion / (self.max_iter)
        para_cs = cs_ratio / 100.0 * 2  #so that max(para_cs) == 1
        
        para = torch.tensor([[para_cs, 
                              para_total_recursions, 
                              para_cur_recursion]])
        return para.float()
    

    def forward(self, x, Phi, block_size=33, feedback=None):
        
        if self.force_no_feedback:
            feedback = np.array(self.feedback)*0 + 1
        elif feedback is None:
            feedback = self.feedback
        else:
            assert len(feedback) == self.LayerNo

        batch_x = x
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
                x = self.fcs[i](x, PhiTPhi, PhiTb, cond, block_size)
                outputs[i].append(x)
        outputs[-1][-1] = x
        
        return outputs
    
#%% Initialize the model

feedback_random = True

feedback = [1] * layer_num
layer_num = len(feedback)
feedback_max = [IPL] * layer_num

feedback_test = feedback_max
group_num = feedback
block_size=[33,33]

if RFMU:
    model = R_COAST(layer_num, feedback=feedback, IPL=IPL, block_size=block_size, nf=32)
else:
    model = COAST(layer_num, feedback=feedback, block_size=block_size, nf=32)

# if model_name.lower() == 'coast':
#     model = COAST(layer_num, feedback=feedback, block_size=block_size, nf=32)
# elif model_name.lower() == 'r-coast' or model_name.lower() == 'r_coast':
#     model = R_COAST(layer_num, feedback=feedback, IPL=IPL, block_size=block_size, nf=32)

model_dir = f"./{args.model_dir}/{model_name}_layer_{layer_num}_IPL_{IPL}_lr_{learning_rate:.2e}_nrtrain={nrtrain_prt}"
model_dir_best_psnr = f"./{args.model_dir}/{model_name}_layer_{layer_num}_IPL_{IPL}_lr_{learning_rate:.2e}_nrtrain={nrtrain_prt}/Best_PSNR"
model_dir_best_ssim = f"./{args.model_dir}/{model_name}_layer_{layer_num}_IPL_{IPL}_lr_{learning_rate:.2e}_nrtrain={nrtrain_prt}/Best_SSIM"

log_file_name = f"{model_dir}/Log"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(model_dir_best_psnr):
    os.makedirs(model_dir_best_psnr)

if not os.path.exists(model_dir_best_ssim):
    os.makedirs(model_dir_best_ssim)

nrtrain = Training_labels.shape[0]

print_flag = True

if print_flag:
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{layer_num} layers')
    print(f'{feedback_max} recursions')
    if feedback_random:
        print(f'Training FLOPS equivalant to {(sum(feedback)+sum(feedback_max))/2:.2f} layers')
    else:
        print(f'Training FLOPS equivalant to {sum(feedback_max)} layers')
    print(f'Max FLOPS equivalant to {sum(feedback_max)} layers')
    print(f'Test FLOPS equivalant to {sum(feedback_test)} layers')

if (platform.system() == "Windows"):
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size,
                             num_workers=0, shuffle=True)
else:
    rand_loader = DataLoader(dataset=RandomDataset(Training_labels, nrtrain), batch_size=batch_size,
                             num_workers=num_workers, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =============================================================================
#%% Load the weights of the model (if start_epoch != 1)
# =============================================================================
# model_path = ''
# optim_path = ''
# set11_loss = ''

# model.load_state_dict(torch.load(model_path))
# optimizer.load_state_dict(torch.load(optim_path))

#THE BELOW LINES WERE NOT TESTED!
if gpu_count > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

psnr_loss, ssim_loss = show_image(model, Phi, [10], args, batch_size=1024*1, 
                                  img_no='all', test_name=validation_name, 
                                  model_name=model_name, feedback=feedback_max)

# =============================================================================
#%% Training init
# =============================================================================
plotEp = 1

if start_epoch == 0:
    loss       = np.zeros(end_epoch-start_epoch)
    set11_loss = np.zeros(((end_epoch-start_epoch)//plotEp, 2))
    idx = -1
else:
    loss       = np.zeros(end_epoch-start_epoch)
    set11_loss = np.vstack([set11_loss, np.zeros(((end_epoch-start_epoch)//plotEp, 2))])
    idx = -1

layer_layout = [1, layer_num-2, 1]
if feedback_random:
    rng = default_rng(seed=seed)

# =============================================================================
#%% Training loop
# =============================================================================
start_time = time()
epoch_time = time()
completion_time = 'Complete_one_epoch_first'

for epoch_i in range(start_epoch + 1, end_epoch + 1):
    print('\n', '-' * 15, 'Epoch: [%d/%d]' % (epoch_i, end_epoch), '-' * 15)
    total_iter_num = np.ceil(nrtrain / batch_size)
    counter = 0
    pbar = tqdm(enumerate(rand_loader), total=total_iter_num)
    for _, data in pbar:
        counter += data.shape[0]
        batch_x = data
        batch_x = batch_x.to(device)
        rand_Phi_index = np.random.randint(rand_num * 1)

        rand_cs_ratio = np.random.choice(train_cs_ratio_set)
        cur_Phi = Phi[rand_cs_ratio][rand_Phi_index].to(device)
        
        x_input = batch_x
        if feedback_random:
            feedback_tr = [rng.choice(np.arange(1,IPL+1), 
                                      size=layer_layout[i], 
                                      replace=True) for i in range(min(layer_num,3))]
            feedback_tr = np.hstack(feedback_tr)
        else:
            feedback_tr = feedback_test
            
        with torch.cuda.amp.autocast(): # Casts operations to mixed precision
            outputs = model(x_input, cur_Phi, feedback=feedback_tr)
            x_output = outputs[-1][-1]
            
            loss_discrepancy = torch.mean(torch.pow(x_output - batch_x, 2))
            loss_all = loss_discrepancy
        
        optimizer.zero_grad()
        scaler.scale(loss_all).backward()
        scaler.step(optimizer)
        scaler.update()
        
        loss_print = torch.mean(torch.pow(x_output - batch_x, 2))
        loss[epoch_i-start_epoch-1] += loss_print * (data.shape[0]/nrtrain)
        
        comp_time = f'Comp.Time={completion_time}'
        elapsed_time = f'Time={timer(start_time)}'
        
        if epoch_i == start_epoch + 1:
            comp_time = f'fb={feedback_tr}'
            output_data = "[%03d/%03d] RMSE=%.2e  %s  %s" % (epoch_i, end_epoch, np.sqrt(loss[epoch_i-start_epoch-1]*nrtrain/counter), comp_time, elapsed_time)
        else:
            output_data = "[%03d/%03d] RMSE=%.2e  %s  %s" % (epoch_i, end_epoch, np.sqrt(loss[epoch_i-start_epoch-1]*nrtrain/counter), comp_time, elapsed_time)
        pbar.set_description(output_data)
    
    #Plot the results
    if epoch_i % plotEp == 0:
        # Plot the training loss
        x_axis = [i+start_epoch for i in range(epoch_i-start_epoch)]
        plt.figure(num=1, dpi=300, figsize=(2*5.5, 1*5.5))
        plt.subplot(1,2,1)
        plt.plot(x_axis, loss[:epoch_i-start_epoch]**0.5, marker='o')
        plt.title('RMSE')
        plt.xlabel('Epochs / {} Training samples'.format(nrtrain))
        plt.grid()
        
        plt.subplot(1,2,2)
        plt.plot(x_axis, 20*np.log10(1+loss[:epoch_i-start_epoch]**0.5), marker='o')
        plt.title('Logarithmic RMSE')
        plt.xlabel('Epochs / {} Training samples'.format(nrtrain))
        plt.ylabel('dB')
        plt.grid()
        plt.suptitle('Learning Curves')
        
        # Plot the training loss for the last "gap" epochs only (the word gap is misleading):
        gap = 50
        if epoch_i-start_epoch > gap:
            x_axis = [i+1 for i in range(epoch_i-start_epoch)]
            plt.figure(num=2, dpi=300, figsize=(2*5.5, 1*5.5))
            plt.subplot(1,2,1)
            plt.plot(x_axis[max(0,epoch_i-start_epoch-gap):epoch_i-start_epoch], 
                        loss[max(0,epoch_i-start_epoch-gap):epoch_i-start_epoch]**0.5, marker='o')
            plt.title('RMSE')
            plt.xlabel('Epochs / {} Training samples'.format(nrtrain))
            plt.grid()
            
            plt.subplot(1,2,2)
            plt.plot(x_axis[max(0,epoch_i-start_epoch-gap):epoch_i-start_epoch], 
                      20*np.log10(1+loss[max(0,epoch_i-start_epoch-gap):epoch_i-start_epoch]**0.5), marker='o')
            plt.title('Logarithmic RMSE')
            plt.xlabel('Epochs / {} Training samples'.format(nrtrain))
            plt.ylabel('dB')
            plt.grid()
            plt.suptitle('Learning Curves')
            
        plt.show()
        
        # Compute the validation loss
        idx += 1
        psnr_loss, ssim_loss = show_image(model, Phi, cs_ratios=[10], args=args, batch_size=1024*1, img_no='all', test_name=validation_name, model_name=model_name, feedback=feedback_test, disp_image=False)
        set11_loss[idx]      = psnr_loss.mean(), ssim_loss.mean()
        
        # Plot the validation PSNR
        x_axis = (1+np.arange(idx+1))*plotEp
        # x_axis = start_epoch+(1+np.arange(idx+1))*plotEp
        plt.figure(num=3, dpi=300, figsize=(2*5.5, 1*5.5))
        plt.subplot(1,2,1)
        # plt.plot(x_axis, my_loss[:idx+1, 0],    marker='o', label='Custom Dataset')
        plt.plot(x_axis, set11_loss[:idx+1, 0], marker='o', label='Set11 Dataset')
        plt.title('PSNR')
        plt.xlabel('Epochs / {} Training samples'.format(nrtrain))
        plt.ylabel('dB')
        plt.grid()
        plt.legend()
        
        # Plot the validation SSIM
        plt.subplot(1,2,2)
        # plt.plot(x_axis, my_loss[:idx+1, 1],    marker='o', label='Custom Dataset')
        plt.plot(x_axis, set11_loss[:idx+1, 1], marker='o', label='Set11 Dataset')
        plt.title('SSIM')
        plt.xlabel('Epochs / {} Training samples'.format(nrtrain))
        plt.grid()
        plt.legend()
        plt.suptitle('Learning Curves')
        
        plt.show()
    
    duration = time() - epoch_time
    time_per_epoch = duration / (epoch_i-start_epoch)
    total_epochs = (end_epoch - start_epoch)
    completion_time = estimate(time_per_epoch * total_epochs)
    
    #Logs        
    output_data = str(datetime.now()) + " [%d/%d] Total loss: %.4f, discrepancy loss: %.4f\n" % (
        epoch_i, end_epoch, loss_all.item(), loss_discrepancy.item())
    write_data(log_file_name, output_data)
    
    if set11_loss[:idx+1,0].max() == set11_loss[idx,0]:
        checkpoint_path  = f"./{model_dir_best_psnr}/epoch={epoch_i}_PSNR_SSIM={set11_loss[:idx+1,0].max():.2f}_{set11_loss[:idx+1,1].max():.4f}_IPL={IPL}_rngFB={feedback_random}_checkpoint.pkl"
        checkpoint = {"model": model.cpu().state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(),
                      "loss": set11_loss}
        torch.save(checkpoint, checkpoint_path)
        model.to(device)
    
    if set11_loss[:idx+1,1].max() == set11_loss[idx,1]:
        checkpoint_path  = f"./{model_dir_best_ssim}/epoch={epoch_i}_PSNR_SSIM={set11_loss[:idx+1,0].max():.2f}_{set11_loss[:idx+1,1].max():.4f}_IPL={IPL}_rngFB={feedback_random}_checkpoint.pkl"
        checkpoint = {"model": model.cpu().state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(),
                      "loss": set11_loss}
        torch.save(checkpoint, checkpoint_path)
        model.to(device)

    if epoch_i % save_cycle == 0:
        checkpoint_path  = f"./{model_dir}/epoch={epoch_i}_PSNR_SSIM={set11_loss[:idx+1,0].max():.2f}_{set11_loss[:idx+1,1].max():.4f}_IPL={IPL}_rngFB={feedback_random}_checkpoint.pkl"
        checkpoint = {"model": model.cpu().state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(),
                      "loss": set11_loss}
        torch.save(checkpoint, checkpoint_path)
        model.to(device)

torch.cuda.empty_cache()  #Release GPU memory from cache