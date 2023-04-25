#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

# start = time.time()

def timer(start):
    duration = time.time() - start
    
    hours = duration // 3600
    duration -= hours * 3600
    
    minutes = duration // 60
    duration -= minutes * 60
    
    seconds = duration
    
    hours   = str(int(hours)  ).zfill(2)
    minutes = str(int(minutes)).zfill(2)
    seconds = str(int(seconds)).zfill(2)
    # print('{:.3f} seconds'.format((time.time() - start)/60))
    out = '{}:{}:{}'.format(hours, minutes, seconds)
    # print(out)
    return out
    
def estimate(duration):
    # duration = time.time() - start
    hours = duration // 3600
    duration -= hours * 3600
    
    minutes = duration // 60
    duration -= minutes * 60
    
    seconds = duration
    
    hours   = str(int(hours)  ).zfill(2)
    minutes = str(int(minutes)).zfill(2)
    seconds = str(int(seconds)).zfill(2)
    # print('{:.3f} seconds'.format((time.time() - start)/60))
    out = '{}:{}:{}'.format(hours, minutes, seconds)
    # print(out)
    return out