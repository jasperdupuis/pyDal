# -*- coding: utf-8 -*-
"""
implement the block diagram from Gardner (1993)
"""


import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#import pydrdc
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures

import data_methods

FS = 25600
N = 2**20 # ~1e6 samples

keys,time_dict = data_methods.load_data_time()
keys,spec_dict = data_methods.load_data_spec()
data = time_dict[keys[3]]
x = data[2*N:3*N]
t = np.arange(len(x))/FS


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Not a direct equation reference, but returns
    the band-passed x_b(t) with del_f centered at f of x(t)
    
    also returns the sos object for plotting, if desired:
    # w,h = signal.sosfreqz(sos,1024*8)
    # plt.plot(w,np.abs(h))
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype='band', output='sos')
    y = signal.sosfilt(sos, data)
    return y,sos

freqs = np.arange(500,1000)
alphas = [120/10,140/10]
results = np.zeros((len(alphas),len(freqs)),dtype=np.complex128)
bw = .2

col_index = 0
for fc in freqs:
    row_index = 0
    for alpha in alphas:    
        exp_downshift = np.exp(-1j*np.pi*alpha*t)     # translates down by alpha
        exp_upshift = np.exp(-1j*np.pi*alpha*t)       # translates up by alpha
        
        x_upper , _ = butter_bandpass_filter(x*exp_downshift,
                                             fc-bw/2,
                                             fc+bw/2,
                                             FS)
        x_lower , _ = butter_bandpass_filter(x*exp_upshift,
                                             fc-bw/2,
                                             fc+bw/2,
                                             FS)
        
        x_mult = x_upper*np.conj(x_lower)
        time_ave = np.sum(x_mult)/np.max(t)
        results[row_index,col_index] = time_ave
        row_index = row_index + 1
    col_index = col_index+1
    
plt.figure()
for index in range(results.shape[1]):
    plt.plot(freqs,np.log10(np.abs(results[index,:])),label=freqs[index])
plt.legend()

# # z = (1/np.max(t))*x_upper * np.conj(x_lower)

# zz = (1/np.max(t)) * np.convolve(np.real(x_upper),np.real(x_lower))

# plt.plot(np.abs(zz))
