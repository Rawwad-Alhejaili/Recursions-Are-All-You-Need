#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Thong Nguyen @ StackOverflow. Modified by Rawwad Alhejaili
   link: https://stackoverflow.com/a/62508086
"""

from prettytable import PrettyTable
from copy import deepcopy

def count_params(model, prt='total_params'):
    m = deepcopy(model)  #try copying it to the CPU later
    try:
        for param in m.parameters():
            param.requires_grad = True
    except:
        for param in m.module.parameters():
            param.requires_grad = True
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in m.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    if prt == 'total_params':
        print(f"Total Trainable Params: {total_params}")
    else:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params
