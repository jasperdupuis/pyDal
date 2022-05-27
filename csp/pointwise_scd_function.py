# -*- coding: utf-8 -*-
"""
implement the block diagram from Gardner (1993)

in progress.
"""

import time
from multiprocessing import Pool, freeze_support

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import data_methods

FS = 25600
N = 2**16 # ~1e6 samples

keys,time_dict = data_methods.load_data_time()
keys,spec_dict = data_methods.load_data_spec()
data = time_dict[keys[3]]
x1 = data[2*N:3*N]
x2 = data[3*N:4*N]
x3 = data[4*N:5*N]


t = np.arange(len(x1))/FS

freqs = np.arange(1000,1100,10)
# freqs = [500, 1000]
alphas = np.arange(60,150)/10
# alphas = [14]
bw = 1



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

def calculate_scd(p_x,
                  p_t=t,
                  p_fs=FS,
                  p_bw=bw,
                  p_freqs=freqs,
                  p_alphas=alphas):
    results = np.zeros((len(p_alphas),len(p_freqs)),dtype=np.complex128)
    col_index = 0
    for fc in p_freqs:
        row_index = 0
        for alpha in p_alphas:    
            exp_downshift = np.exp(-1j*np.pi*alpha*p_t)     # translates down by alpha
            exp_upshift = np.exp(-1j*np.pi*alpha*p_t)       # translates up by alpha
            
            x_upper , _ = butter_bandpass_filter(p_x * exp_downshift,
                                                 fc - p_bw / 2,
                                                 fc + p_bw / 2,
                                                 p_fs)
            x_lower , _ = butter_bandpass_filter(p_x * exp_upshift,
                                                 fc - p_bw / 2,
                                                 fc + p_bw / 2,
                                                 p_fs)
            
            x_mult = x_upper*np.conj(x_lower)
            time_ave = np.sum(x_mult)/np.max(t)
            results[row_index,col_index] = time_ave
            row_index = row_index + 1
        col_index = col_index+1
    return results



# plt.figure()
# for index in range(results.shape[1]):
#     plt.plot(alphas,np.abs(results[:,index]),label=freqs[index])
# plt.legend()


if __name__=='__main__':
    freeze_support()
    start = time.time()    
    result = calculate_scd(x1,t,FS,bw,freqs,alphas)
    end = time.time()
    print('single unthreaded: ' + str(end - start))
    
    start = time.time()
    with Pool(10) as p:
        inputs = [x1,x2,x3]
        results = p.map(calculate_scd, inputs)    
    end = time.time()
    print('three multithreaded: ' + str(end - start))
  
    
  
    
