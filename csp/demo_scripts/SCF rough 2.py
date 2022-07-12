# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:18:02 2022

@author: Jasper
"""

import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt

import sys
sys.path.append('..\..\csp')
import util_signals

def reshape_with_overlap(data,
                         N,
                         N_p):
    """
    builds overlapped data array 
    """
    n_cols = int ( N_p )
    n_rows = int( N )
    result = np.zeros((n_rows,n_cols))
    for n in np.arange(n_cols):
        low = n 
        high = low + N
        result[:,n] = data[low:high]
    return result


num_seconds = 5
fs = 1e4
t = np.arange(num_seconds*fs)/fs
fc = 1e3
bit_rate = 100
s_t = util_signals.random_BPSK(t,fc,bit_rate,0.01)



ext = [-10,10,min(f),max(f)]
plt.imshow(
    np.log10(np.abs(S_X_alpha)),
    aspect='auto',
    extent=ext);plt.colorbar()    
    
plt.imshow(np.abs(S_X_alpha),aspect='auto');plt.colorbar()    


    
    
