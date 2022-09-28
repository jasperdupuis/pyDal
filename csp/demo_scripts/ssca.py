# -*- coding: utf-8 -*-
"""

Demonstrate and plot the SSCA algorithm with some data sets:
    AM
    BPSK
    An accelerometer data set

Compare these with a PSD for the same data.

"""

import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import sys
sys.path.append('../../csp')
import strip_spectral_correlation_analyzer as pySSCA
import util_signals
import util_plotting
import data_methods

CALC_SCF = False
GRAPH_SCF = True

def estimate_psd(s_t,fs,N_p,L):
    """
    Look at the PSD of signal with same window and N_p
    """
    f,t,XFT = signal.stft(s_t,
        fs=fs,
        window=np.hanning(N_p),
        nperseg=N_p,
        noverlap=N_p-L,
        return_onesided=False)
    XFT2 = np.abs(XFT)**2
    psd = np.mean(XFT2,axis=1)
    return psd    


if __name__ == "__main__":
    
    num_seconds = 1.1
    fs = 25600
    t = np.arange(num_seconds*fs)/fs    
    
    signals = dict()
    results = dict()
    #HDW accel signal
    # keys[3] == 7 knot accel Dec (day 1) #
    keys,time_dict = data_methods.load_data_time()
    keys,spec_dict = data_methods.load_data_spec()
    data = time_dict[keys[3]]
    signals['HDW, 7kts Day 1 (Dec)'] = data[10*fs:int((10+num_seconds))*fs]
    # textbook signals
    signals['BPSK, fc=10k fb=500'] =  s_t = util_signals.random_BPSK(t,
                                    p_fc=1e4,
                                    p_bit_rate=500,
                                    p_noise_power=2 )
    # signals['Noise']= util_signals.random_noise(t)
    signals['AM, fc=5k fm=250']= util_signals.amplitude_modulation(t,
                                            p_fc=5e3,
                                            p_fm=2.5e2,
                                            p_noise_power = 2)        
    N_p = fs//2 # for symmetry reasons only. may be slower.
    L_factor = N_p
    L = N_p//L_factor

    for key,value in signals.items():
        _,_,S_X = pySSCA.SSCA_April_1994(value,
                            FS=fs,
                            N_p=N_p,
                            L_factor=L_factor)
        surface = np.log10(np.abs(S_X))
        
        P = S_X.shape[0]
        df = 1/N_p
        da = 1/P
        dt = P
        product_df_dt = df*dt
        
        
        scf = ndimage.rotate(surface, angle=-45,reshape='True')#,mode='wrap')
        psd = estimate_psd(value,fs,N_p,L)
    
        results[key + ' PSD'] = psd
        results[key + ' SCF'] = scf
        print(key)
        
    freqs = np.fft.fftfreq(N_p,1/fs)
    alphas =     np.arange(-N_p,N_p)*fs/N_p
    nrow = 2
    ncol = 3
    selectors = []
    for row in range(nrow):
        for col in range(ncol):
           selectors.append((row,col)) 
    f,ax_arr = plt.subplots(nrow,ncol,figsize=(15,9))
    ind_psd = 0
    ind_scf = 0
    for key,value in results.items():
        if 'SCF' in key: #(SCF)
            label = key
            _,im = util_plotting.plot_and_return_2D_axis(
                ax_arr[selectors[ind_scf+3]],
                value,
                alphas/(2*fs),
                alphas/fs,
                p_x_min = -1e6,
                p_x_max = 1e6,
                p_y_min = -1e6,
                p_y_max = 1e6,
                p_label = label,
                p_linear=True)
            ind_scf += 1
        else: # (PSD)
            label = key
            _,im = util_plotting.plot_and_return_1D_axis(
                ax_arr[selectors[ind_psd]],
                value  ,  
                freqs,
                p_x_min = min(freqs),
                p_x_max = max(freqs),
                p_label = label,
                p_linear=False)    
            ind_psd += 1
    # cbar = f.colorbar(im, ax=ax_arr.ravel().tolist())
    # cbar.ax.set_ylabel('arbitrary')
    