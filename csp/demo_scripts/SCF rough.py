# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:55:26 2022

@author: Jasper
"""

# run orca_accels first to get data in to run_data

import numpy as np
from data_methods import butter_bandpass_filter as bpf
import matplotlib.pyplot as plt

import sys
sys.path.append('..\..\csp')
import dft_alphas

keys = list(run_data.keys())
sensor = 'H2'

x_total = run_data[keys[6]][sensor]

# N = param.FS * 3 #3s of accel data.
N = 1024

x = x_total[10*N:11*N]
t = np.arange(len(x))/param.FS

ntau = N//2
taus = np.arange(ntau)*2
alpha_min = 0
alpha_max = 20
alpha_resolution = 0.1 #gets factored in to freq basis next
freqs = np.fft.fftfreq(int(param.FS / alpha_resolution),
                       1/param.FS)[:N//2] # all freqs
alphas_max_index = freqs - alpha_max < 0
alphas_min_index = freqs - alpha_min > 0 
selection = np.logical_and(alphas_max_index,alphas_min_index)
alphas = freqs[selection]
alphas_pos = alphas
alphas_pos_neg = np.concatenate((-1*alphas[::-1], alphas))

#
#
# "Cyclic autocorrelation function"
# Antoni 2009 equation 38-41, 
# Note change of to to tprime in this implementation
CAF = np.zeros( (len(alphas_pos),len(taus)),
               dtype = np.complex128)
for index_tau in range(len(taus)): #g[x[n]] = x[n].x[n-tau]
    tau = taus[index_tau]
    x1 = np.roll(x, -1*tau//2) # implement circular shift with given tau.
    x2 = np.roll(x, tau//2) # implement circular shift with given tau.
    xx = x1 * x2 # For the given tau, this is the correlation sequence 
    t_prime = t #- (tau * 1/param.FS)
    d = dft_alphas.partial_DFT(xx,t_prime,alphas_pos)
    CAF[:,index_tau] = d # IAF is now k,tau indexed (k is a discrete freq bin)
extent = [min(taus),
          max(taus),
          min(alphas_pos),
          max(alphas_pos)]
plt.figure()
plt.imshow(
    np.abs(CAF),
    aspect='auto',
    # extent=extent,
    origin='lower')
plt.colorbar()
#
#
# "Instantaneous autocorrelation function"
IAF = np.zeros( (len(taus),len(t)),
               dtype = np.complex128)
for index_tau in range(len(taus)):
    X = CAF[:,index_tau]
    IAF[index_tau,:] = dft_alphas.partial_IDFT(X,t,alphas_pos)
extent = [min(t),
          max(t),
          min(taus),
          max(taus)]
plt.figure()
plt.imshow(
    np.abs(IAF),
    aspect='auto',
    extent=extent,
    origin='lower')
plt.colorbar()
#
#
# Spectral correlation function from CAF
# Antoni 2009 : Eqn 41 and Figure 30
# This doesn't look as expected
SCF = np.zeros_like(CAF)
for alpha_index in range(len(alphas_pos)):
    SCF[alpha_index,:] = np.fft.fft(CAF[alpha_index,:])
frequencies_pos = np.fft.fftfreq(N,1/param.FS)[:N//2]
SCF_chop = SCF[:N//2]
extent = [min(frequencies_pos),
          max(frequencies_pos),
          min(alphas_pos),
          max(alphas_pos)]
plt.figure()
plt.imshow(
    np.abs(SCF),
    aspect='auto',
    # extent=extent,
    origin='lower')
plt.colorbar()


#
#
#
# Try SCF direction Antoni equation 43
# g[x[n]] is the bandpass multiplication
BW = 1
fc = 1000
res = np.zeros(len(alphas_pos),
                   dtype=np.complex128)
for alpha_index in range(len(alphas_pos)):
    f1 = fc - alphas_pos[alpha_index]
    f2 = fc + alphas_pos[alpha_index]
    x1,_ = bpf(x,f1-BW,f1+BW,param.FS)
    x2,_ = bpf(x,f2-BW,f2+BW,param.FS)
    gx = x1*x2
    res = dft_alphas.partial_DFT(gx,t,alphas_pos[alpha_index])





