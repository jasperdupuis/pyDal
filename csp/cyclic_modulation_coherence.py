# -*- coding: utf-8 -*-
"""

Methods from Antoni's 2012 paper 
'Detection of Surface ships from interception of cyclotationary signature with 
cyclic modulation coherence'

There are two methods in this paper:
    
    Cyclic Modulation Spectrum (first in 2009 tutorial paper)
    Cyclic modulation coherence (novel in 2012 paper)

"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

#import pydrdc
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures


def create_spectrogram(
        x,
        fs = 25600,
        window=np.hanning(1024),
        N_prime = 2**10,
        noverlap = 2**8):
    """
    Compute spectrogram magnitude using the signal.stft function
    
    N_prime is length of window
    noverlap is overlap per shift.
    
    # TODO: Verify stft performs window & normalization as expected.
    # TODO: Verify the effect of non-normalized window on outcomes.
    """
    f_gram, t_gram, S_gram = signal.stft(
        x = x, 
        fs = fs,
        window= window,
        nperseg = N_prime, 
        noverlap= noverlap,
        nfft=None, 
        detrend=False,
        return_onesided=True,
        boundary='zeros',
        padded=False,
        axis=- 1)
    #Transpose the gram to align with typical navy view.
    S_gram = S_gram.T
    return f_gram,t_gram,S_gram


def plot_spectrogram(gram,f,t):
    """
    """
    extent = [np.min(f),np.max(f),np.min(t),np.max(t)]
    plt.imshow(10*np.log10(np.abs(gram)**2), 
               extent = extent,
               cmap = 'RdBu',
               aspect = 'auto',
               origin='lower')
    plt.title('Spectrogram (units squared per hz)')
    plt.xlabel('Frequency [Hz], del_f = ' + str(f[1])[:4])
    plt.ylabel('Time [sec], del_t = ' + str(t[1])[:4])
    plt.show()

    
def create_cyclic_modulation_coherence(x,fs,w,N_prime,L):
    """
    Implements equation 18 from Antoni 2012.
    Nests the spectrogram calculation as a subfunction.
    
    x - entire time series to consider
    fs - sample freq
    w - window function (1d array) for sliding STFT
    N_prime - length of w
    R - overlap factor for STFT (in COMMON between adjacent windows)

    Implictly but not used, L is the amount shifted per window,
    i.e. 1-R.
    
    freqs, alphas, cmc = create_cyclic_modulation_coherence(
        s, FS, w, N_prime, L)
    """
    #Passed a given time-series x, calculate the CMC     
    f,t,gram_cmplx = create_spectrogram(x,fs,w,N_prime,L)
    spectrogram = np.abs(gram_cmplx)**2
    I = spectrogram.shape[0]
    cms = np.fft.fft(spectrogram,axis=0) / I
    alphas = np.fft.fftfreq(I,t[1])
    
    #drop negative frequency portions:
    alphas = alphas[:len(alphas)//2]
    cmc_pos_alpha = cms[:I//2,:]
    
    #scale result
    for row in range(cmc_pos_alpha.shape[1]):
        r = cmc_pos_alpha[:,row]
        r = r/r[0]
        cmc_pos_alpha[:,row] = r
    return f,alphas,cmc_pos_alpha

    
def plot_cyclic_modulation_coherence(array,f,alphas,linear=True):
    """
    typical use:
    freqs, alphas, cmc = create_cyclic_modulation_coherence(
        x, FS, w, N_prime, R)    
    plot_cyclic_modulation_coherence(cmc, freqs, alphas,linear=False)
    
    ICMC = np.sum(cmc,axis=1)
    plt.figure()
    plt.plot(alphas[1:],np.abs(ICMC[1:]))
    """
    extent = [np.min(f),np.max(f),np.min(alphas),np.max(alphas)]
    array = np.abs(array)
    if not linear:
        array = 10*np.log10(array)
    plt.imshow(array,
               extent=extent,
               origin='lower',
               aspect='auto',
               cmap='RdBu')
    plt.title('Cyclic Modulation Spectrum')
    plt.xlabel('Frequency [Hz], del_f = ' + str(f[1])[:5])
    plt.ylabel('Alpha [Hz], del_a = ' + str(alphas[1])[:5])
    plt.colorbar()


def create_sampled_CMC_and_gram(
        data_samples,
        fs,
        M,
        w,
        N_prime,
        L,
        n_alphas,
        n_freqs,
        ):
    """
    Given an array of data samples, each of some real time duration,
    produce a the 3D PSD and CMS array (for each).
    
    The three axis are: 
        1: major sample window index, 
        2: n_time (gram) / n_alphas (CMC)
        3: n_freqs (gram and CMC)
    
    This produces CMC array (subfunction), which is CMS array divided 
    by the entry at alpha = 0 for each entry.
    
    # TODO: Does it matter if average then CMC, or CMC then average?
    # TODO: (MATH)
    
    data_samples
    fs
    M - total number of expected windows in the gram STFT
    w - window function, 1d array
    N_prime - length of window function
    L - window shift per step
    n_alphas - number of expected cyclic frequencies
    n_freqs - number of expeceted spectrogram frequencies
    
    """
    results_cmc = np.zeros((data_samples.shape[1],n_alphas,n_freqs))
    results_spec = np.zeros((data_samples.shape[1],M//2,n_freqs))
    freqs = np.zeros(n_freqs)
    alphas = np.zeros(n_alphas)
    for index in range(data_samples.shape[1]):
        freqs,t,temp = \
            create_spectrogram(data_samples[:,index],fs,w,N_prime,L)
        results_spec[index,:,:] = np.abs(temp)[:M//2,:]
        
        freqs,alphas,temp = \
            create_cyclic_modulation_coherence(
                data_samples[:,index], fs, w, N_prime, L)
        results_cmc[index,:,:] = np.abs(temp)
    
    return t,freqs,alphas,results_cmc,results_spec


def calculate_parameters(
        N,
        fs,
        del_f,
        overlap_minor):
    # used parameters, calculate from above values:
    N_prime = int( fs / del_f )
    L = int(N_prime * (1 - overlap_minor)) # window shift per step, Trevorrow2021 calls this R
    M = int ( ( ( N - N_prime ) // (N_prime-L)) + 1 ) # number of time-entries in spectrogram
    dt = 1/fs
    del_alpha_achieved = (M *dt * L) **-1
    n_freqs = N_prime // 2  + 1 # positive plus DC at 0
    n_alphas = ( M + 1 ) // 2
    # I as defined in Antoni is retrieved from the STFT dimensions
    w = np.hanning(N_prime)
    
    return N_prime,L,M,del_alpha_achieved,n_freqs,n_alphas,w
