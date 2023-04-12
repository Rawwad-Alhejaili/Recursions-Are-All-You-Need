#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from tqdm import tqdm

# ratio_dict = {1: 10, 4: 43, 10: 109, 20: 218, 25: 272, 30: 327, 40: 436, 50: 545}
N = 1089  #Moved below
total_phi_num = 50

train_cs_ratio_set = [1, 4, 5, 10, 20, 25, 30, 40, 50]
# train_cs_ratio_set = [10]

Phi_all = {}
for cs_ratio in tqdm(train_cs_ratio_set):
    M = int(np.round(N*cs_ratio/100))
    # Phi_all[cs_ratio] = np.zeros(total_phi_num, M, N)
    C = np.random.normal(size=(total_phi_num,M,N))
    # Phi_name = './%s/phi_sampling_%d_%dx%d.npy' % (args.matrix_dir, total_phi_num, size_after_compress, 1089)
    # Phi_data = np.load(Phi_name)
    for k in range(total_phi_num):
        C[k] = scipy.linalg.orth(C[k].T).T
        assert np.round((C[k] @ C[k].T - np.eye(M))**2, 10).sum() == 0 #Check that C is unitary
    
    Phi_all[cs_ratio] = C
    
    np.save(f'sampling_matrix/phi_sampling_{total_phi_num}_{M}x{N}', C)

# C = np.random.normal(size=(M,N))
