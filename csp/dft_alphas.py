# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 11:08:15 2022

@author: Jasper
"""

import numpy as np

def partial_DFT(x,t,freq_basis):
    """
    Compute the DFT using a provided frequency basis, instead of the default
    relationship between frequency and time series length
    (unless alpha_basis is the same as np.fft.fftfreq(len*x),1/FS)
    x - the time series
    t - the time basis corresponding to x
    freq_basis - the target frequencies over which the partial DFT wil be computed
    """
    # non-pythonic loop method
    # res = np.zeros(len(freq_basis),dtype=np.complex128)
    # for entry in range(len(res)):
    #     freq = freq_basis[entry]
    #     exp = np.exp(-2j*np.pi*t*freq)
    #     res[entry] = np.sum(x*exp)
    # pythonic method using broadcasting and np.sum
    arr = np.dot(freq_basis[:,None],t[None,:])
    exps = np.exp(-2j*np.pi*arr)
    broadcast = x * exps
    res = np.sum(broadcast,axis=1)

    return res
        
def partial_IDFT(X,t,freq_basis):
    """
    Compute the IDFT over the passed frequency basis instead of default values
    
    X is capitalized on purpose - is not a time series!
    """
    #non-pythonic method
    # res = np.zeros(len(t),dtype=np.complex128) 
    # for freq_index in range(len(freq_basis)):
    #     res = \
    #         res \
    #         + (\
    #            X[freq_index] \
    #            * np.exp(2j*np.pi*t*freq_basis[freq_index])
    #            )
    #pythonic method using broadcasting and np.sum
    arr = np.dot(t[:,None],freq_basis[None,:])
    exps = np.exp(2j*np.pi*arr)
    broadcast = X * exps
    res = np.sum(broadcast,axis=1)
    
    return res
  