# -*- coding: utf-8 -*-
"""
Strip spectrum analyzer. Incomplete.

"""


import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt

#import pydrdc
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures


FS = 25600
N = 2**14 # Total data length to consider.
T_s = 1/FS # Sample period
N_prime = (2**10) # Total sub-window length (STFT hanning, e.g.)
T = T_s * N_prime # Sub-window length in time domain.
del_f = FS/N_prime # frequency resolution of first FFT (STFT hanning e.g.)
del_a = del_f # NOT THE SAME AS del_alpha! i.e. these are bandwidth of input filters
del_t = N*T_s # Total data length in time domain.


from data_methods import load_data_time,load_data_spec
keys,time_dict = load_data_time()
keys,spec_dict = load_data_spec()
data = time_dict[keys[4]]
x = data[N:2*N]
t = np.arange(len(x))/FS


def reshape_with_overlap(data,
                         n_length,
                         n_overlap):
    """
    builds overlapped data array 
    """
    n_cols = int ( (len(data) // n_overlap) - n_length // n_overlap + 1 )
    n_rows = int( n_length )
    result = np.zeros((n_rows,n_cols))
    for n in np.arange(n_cols):
        low = n * n_overlap
        high = low + n_length
        result[:,n] = data[low:high]
    return result    


array = reshape_with_overlap(x,int(N_prime),int(L))
P = int(array.shape[1]) # number of stft fourier transforms
w_t = np.hanning(array.shape[0])
arr_w = (array.T *w_t ).T
arr_fft = np.fft.fft(array,axis=0)
arr_freqs = np.fft.fftfreq(array.shape[0],1/FS)
arr_alphas = arr_freqs
