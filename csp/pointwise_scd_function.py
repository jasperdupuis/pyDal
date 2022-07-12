# -*- coding: utf-8 -*-
"""
implement the block diagram from Gardner (1993)

in progress.
"""

import numpy as np
from scipy import signal

import parameters as param


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
                  p_fs=param.FS,
                  p_bw=param.bw,
                  p_freqs=param.scd_point_freqs,
                  p_alphas=param.scd_point_alphas):
    t = np.arange(len(p_x))/p_fs
    results = np.zeros((len(p_alphas),len(p_freqs)),dtype=np.complex128)
    col_index = 0
    for fc in p_freqs:
        row_index = 0
        for alpha in p_alphas:
            exp_downshift = np.exp(-2j*np.pi*alpha*t)     # translates down by alpha
            exp_upshift = np.exp(-2j*np.pi*alpha*t)       # translates up by alpha
            
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


    
  
    
