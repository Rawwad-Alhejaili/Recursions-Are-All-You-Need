#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from tqdm import tqdm

def generate_Gaussian_matrix(N=1089, cs_ratio_set=[10,20,30,40,50], 
                             total_phi_num=50):
    Phi_all = {}
    for cs_ratio in tqdm(cs_ratio_set):
        # Generate the Gaussian sampling matrices
        M = int(np.round(N*cs_ratio/100))
        C = np.random.normal(size=(total_phi_num,M,N))
        
        # Orthogonalize the rows of the sampling matrices
        for k in range(total_phi_num):
            C[k] = scipy.linalg.orth(C[k].T).T
            # Check that C is indeed "unitary"
            assert np.round((C[k] @ C[k].T - np.eye(M))**2, 10).sum() == 0
        
        Phi_all[cs_ratio] = C
        
        # np.save(f'sampling_matrix/phi_sampling_{total_phi_num}_{M}x{N}', C)
        
        return Phi_all
