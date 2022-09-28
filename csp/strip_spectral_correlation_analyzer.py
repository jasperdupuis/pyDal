# -*- coding: utf-8 -*-
"""
Strip spectrum analyzer. Incomplete.

"""


import numpy as np
from scipy import signal
from scipy import ndimage
import matplotlib.pyplot as plt
import time

import util_signals


# FS = 25600
# N = 2**18 # Total data length to consider.
# T_s = 1/FS # Sample period
# N_prime = (2**10) # Total sub-window length (STFT hanning, e.g.)
# T = T_s * N_prime # Sub-window length in time domain.
# del_f = FS/N_prime # frequency resolution of first FFT (STFT hanning e.g.)
# del_a = del_f # NOT THE SAME AS del_alpha! i.e. these are bandwidth of input filters
# del_t = N*T_s # Total data length in time domain.


# from data_methods import load_data_time,load_data_spec
# keys,time_dict = load_data_time()
# keys,spec_dict = load_data_spec()
# data = time_dict[keys[4]]
# x = data[N:2*N]
# t = np.arange(len(x))/FS


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


def SSCA_April_1994(x,
                    FS=1e4,
                    N_p=2**10,
                    L_factor=16):#Must be greater than 4

    """
    Implement the strip spectral correlation analyzer
    matrix method from Eric April's DREO report. Still need to map
    results to the alpha-f plane.
    
    20220708
    Outputs match the Carter method, but no STFT hopping allowed.
    i.e. N_p == L_factor is a requirement right now 
    
    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    N : TYPE
        DESCRIPTION.
    N_prime : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    N = len(x)
    L = N_p // L_factor # shift, 4
    P = ( N - N_p ) // L # Total time-entries in stft, number of constant-time entries
    overlap = N_p - L
    if overlap == 0: overlap = 1
    a_r = np.hanning(N_p) #spectral window
    g_s = np.hanning(P)   #cyclic window, if needed
    f,t,XA_T = signal.stft(x,
                          fs=FS,
                          window=a_r,
                          nperseg=N_p,
                          noverlap=overlap,
                          return_onesided=False)
    XA_T = np.fft.fftshift(XA_T,axes=0)
    XA_T = XA_T[:,:P] 
    XA_T = XA_T.T # nrows = P, ncols = N_p, matches April's paper now.
    
    # Step 3
    # Apply phase correction - or if you prefer, the baseband demodulation.
    X_phase = np.zeros_like(XA_T)
    spec_freqs_norm = np.arange(-N_p//2,N_p//2) 
    phases = np.exp(-2j*np.pi*spec_freqs_norm/N_p)
    for n in range( P ): # n is the time entry index
        X_phase[n,:] = phases**n    
    X_g = XA_T * X_phase

    # Step 3 (still)
    # apply the multiplication and cyclic window 
    # As in equation 22
    # This is where the autocorrelation operation comes in
    
    # This is old, for hopping STFT operation.
    # To do a hopping STFT must also decimate x[n] 
    # So, leave commented out for now.
    # for n in range(P): # n is the time entry index
    #     start = n * L
    #     end = start + N_p
    #     xc = np.conj(x[start:end])
    #     try:
    #         X_g[:,n] = X_g[:,n] * xc * g_s
    #     except:
    #         print("Broke at correlation multiplcation")
    
    # New 20220708
    # see if this works with non-hopping stft (e.g. 1 shift per transform)
    # 20220712 It does!
    xsel = x[:P].reshape(P,1)
    ones = np.ones_like(XA_T)
    x_t_conj = np.conj(xsel*ones)
    X_g = X_g * x_t_conj
    
    # Step 4
    S_X = np.fft.fft(X_g,axis=0)
    S_X = np.fft.fftshift(S_X,axes=0)
    alphas = np.fft.fftfreq(N,1/FS)
    freqs = np.fft.fftfreq(N_p,1/FS)
    #These also need to be fft shfited.
    freqs = np.fft.fftshift(freqs)
    alphas = np.fft.fftshift(alphas)
    
    return alphas,freqs,S_X
   
def SSCA_April_1994_get_f(k,q, N_p,N):
    k = k - (N_p//2)
    q = q - (N//2)
    return (    ( k / ( 2*N_p ) ) - ( q / ( 2 * N) ) )

def SSCA_April_1994_get_alpha(k,q, N_p,N):
    k = k - (N_p//2)
    q = q - (N//2)
    return ( ( k / N_p  ) + ( q /  N)  )

def SSCA_April_1994_get_k(
        alpha,f, N_p):
    return int( ( ( alpha / 2 ) + f) * N_p  )

def SSCA_April_1994_get_q(
        alpha,f, N):
    return int( ( ( alpha / 2  ) - f ) * N   )

def get_mask_from_maps(freq,fmap,df):
    high = fmap < freq + df /2
    low = fmap > freq - df /2
    sel = np.logical_and(low,high)
    return sel


def return_freq_index(val,hcut,lcut):       
    """
    From passed val and h and l cutoff lists, 
    return index where the value goes on the frequency axis.
    """# pass value, find bin index it belongs in
    hcut = hcut - val > 0
    lcut = lcut - val > 0
    sel = np.logical_xor(hcut,lcut)
    index = np.where(sel==True)    
    index = index[0][0] #peel back array structure around result
    return index


# 20220708 Pretty confident this works as expected,
# but using it beyond the S_X 2d array
# method is not provided in the work. 
def SSCA_Carter_1992(x,
                    FS=1e4,
                    N_p=2**10, # stft window length, number of constant-frequency entries
                    L_factor = 16 ): #must be greater than 4
    # TRY TO FOLLOW CARTER 1992
    # USE THEIR VARIABLE NAMES FOR CLARITY
    N = len(x)
    L = N_p // L_factor # shift, should be less than N_p/4
    P = ( N - N_p ) // L # Total time-entries in stft, number of constant-time entries
    a_r = np.hanning(N_p)
    f,t,X_T = signal.stft(s_t,
                          fs=fs,
                          window=a_r,
                          nperseg=N_p,
                          noverlap=N_p-L,
                          return_onesided=False)
    X_T = X_T[:,:P]
    X_T = np.fft.fftshift(X_T,axes=0)
    phase_corr = np.zeros_like(X_T,dtype=np.complex128)
    for k in range (N_p): # in range of spec freq bins
        for m in range( P ): # in range of time entries
            exp = np.exp(-2j*np.pi*(m*k*L/N_p))  
            phase_corr[k,m] = exp
    X_T_phased = X_T * phase_corr

    X_T_repl = np.zeros((N_p,L*P),dtype=np.complex128)
    one_basis = np.ones((N_p,L),dtype=np.complex128)
    for index in range(P):
        row = np.reshape(X_T_phased[:,index],(N_p,1))
        start = index * L 
        end = ( index + 1 ) * L
        X_T_repl[:,start:end] = row * one_basis
        
    z = X_T_repl * np.conj( x [:L*P] )    
    S_X_alpha = np.fft.fft(z,axis=1) #this is the correct axis (checked)
    # S_X_alpha = np.fft.fftshift(S_X_alpha,axes=1)
    return f, S_X_alpha



if __name__ =='__main__':
    # Set test signal time basis
    num_seconds = 1
    fs = 1e4
    t = np.arange(num_seconds*fs)/fs    
    # signal selection
    # s_t = util_signals.random_BPSK(t,
    #                                 p_fc=2e3,
    #                                 p_bit_rate=200,
    #                                 p_noise_power=2 )
    # s_t = util_signals.random_noise(t)
    s_t = util_signals.amplitude_modulation(t,
                                            p_fc=2e3,
                                            p_fm=2e2,
                                            p_noise_power = 2)
    N_p = 2**12
    L_factor = N_p
    L = N_p//L_factor

    # Look at the PSD of signal with same window and N_p
    # f,t,XFT = signal.stft(s_t,
    #     fs=fs,
    #     window=np.hanning(N_p),
    #     nperseg=N_p,
    #     noverlap=N_p-L,
    #     return_onesided=False)
    # XFT2 = np.abs(XFT)**2
    # psd = np.mean(XFT2,axis=1)
    # plt.figure();plt.plot(f,np.log10(psd))
    
    april_start = time.time()
    _,_,S_X_April = SSCA_April_1994(s_t,
                            FS=fs,
                            N_p=N_p,
                            L_factor=L_factor)
    april_end = time.time()

    carter_start = time.time()
    # _,S_X_Carter = SSCA_Carter_1992(s_t,
    #                             FS=fs,
    #                             N_p=N_p,
    #                             L_factor=L_factor)    
    carter_end = time.time()
    
    print("April method: " + str(april_end - april_start))
    print("Carter method: " + str(carter_end - carter_start))


    # Plot the April result
    extent = [-0.5,0.5,-1,1]    
    ext = []
    for e in extent: 
        ext.append(e*fs)

    surface_April = np.log10(np.abs(S_X_April))
    plt.figure();
    plt.imshow(surface_April,
                aspect='auto',
                extent= ext,
                origin='lower')
    plt.colorbar()
    
    # # Plot the Carter result
    # surface_Carter = np.log10(np.abs(S_X_Carter))
    # plt.figure();
    # plt.imshow(surface_Carter,
    #             aspect='auto',
    #             extent= ext,
    #             origin='lower')
    # plt.colorbar()

    # #work in only normalized units now
    P = S_X_April.shape[0]
    df = 1/N_p
    da = 1/P
    dt = P
    product_df_dt = df*dt
    
    # ME TRYING TO ROLL MY OWN
    # QUESTIONABLE...
    # Build arrays of alpha and f values.    
    
    alpha_map = np.zeros_like(S_X_April,dtype=np.float64)
    freq_map = np.zeros_like(S_X_April,dtype=np.float64)
    
    # # 20220711
    # # BROWN SSCA NOTES
    # # FOR THE KTH OUTPUT FFT,
    # # ZERO FREQUENCY CORRESPONDS TO ALPHA = F_K
    # # TBD: May need to FFTSHIFT later... keep an eye on this.
    # f_list = np.arange(-N_p//2,N_p//2)/N_p  # for the FFTs I do only
    # a_list = np.arange(-P,P)/P              # for the FFTs I do only
    # for k in range(N_p):
    #     fk = f_list[k]
    #     for q in range(P):
    #         alpha_0 = fk + (q - P//2)*da
    #         alpha_map[k,q] = alpha_0            
    #         freq_0= (fk/2) - ( (q-P//2) *da/2)
    #         freq_map[k,q] = freq_0

    #20220712 
    # Try to apply April's formula 24 to generate f, alpha maps
    for k in range(N_p):
        for q in range(P):
            alpha_0 = SSCA_April_1994_get_alpha(k,q,N_p,P)
            alpha_map[q,k] = alpha_0            
            freq_0= SSCA_April_1994_get_f(k,q,N_p,P)
            freq_map[q,k] = freq_0
                 
    # Below plots are the maps calculated above.
    # Shows some confidence in the maps: using NON-TRANSPOSE S_X
    # the zero-spec freq line runs top left to bottom right
    # the zero-cycfreq line runs bottom left to top right
    # After 45 deg rotaitons these end up on the right axes.
    # plt.figure();plt.imshow(freq_map,aspect='auto');plt.colorbar()
    # most positive point in top right corner (Correct)
    # plt.figure();plt.imshow(alpha_map,aspect='auto');plt.colorbar()
    # most positive point in top left corner (Correct)
    
    # Verify the overlap of positive alpha and freq are correct
    # keeping in mind there is a 45 deg CW rotation (of the array)
    # or a 45 deg CCW rotation (of the cartesian basis to f-alpha)
    # to get to biplane
    a_sel = alpha_map > 0
    f_sel = freq_map > 0
    positive_basis = np.logical_and(a_sel,f_sel)
    # plt.figure();plt.imshow(positive_basis,aspect='auto');plt.colorbar()
    # temp = positive_basis * surface_April
    # plt.figure();plt.imshow(temp,aspect='auto')    
    """
    Now I have confidence that I have 
    a spec and cycle  freq value for each S_X element.
    Use a bin algorithm to remap this to a new array... somehow?
    20220712: MASKING WAS WAY TOO SLOW
    Need another way. Stick k and q? They give weird results.
    """

#     n_freq_basis =   N_p
#     n_alpha_basis =   P
#     f_list = \
#         np.arange(-n_freq_basis,n_freq_basis)/(2*n_freq_basis)
#     a_list = \
#         np.arange(-n_alpha_basis,n_alpha_basis)/n_alpha_basis
#     f_h_list = f_list + df/2
#     f_l_list = f_list - df/2
#     a_h_list = a_list + da/2
#     a_l_list = a_list - da/2

#     n_rows = len(a_list)
#     n_cols = len(f_list)
#     results = np.zeros((n_rows,n_cols),dtype=np.float64)
#     for alpha in a_list:
#         a_index = return_freq_index(alpha,a_h_list,a_l_list)
#         for freq in f_list:
#             if np.abs(freq) > np.abs(alpha / 2): 
#                 # Skip if outside the basis area
#                 # This is pretty loose, can tighten if statement up later
#                 continue
#             f_index = return_freq_index(freq,f_h_list,f_l_list)
#             k = SSCA_April_1994_get_k(alpha,freq,N_p) + N_p // 2
#             q = SSCA_April_1994_get_q(alpha,freq,P) + P//2
#             results[a_index,f_index] = surface_April[q,k]
#         print(alpha)
    
    # # r = results
    # # r = np.fft.fftshift(results,axes=0)
    # # ext = [-fs/2,fs/2,-fs,fs]    
    # # plt.figure();plt.imshow(r,
    # #     extent=ext,
    # #     aspect='auto');plt.colorbar()
    
    # # Try to use image rotation libraries to do this,
    # # Appears to work as expected.
    # z = ndimage.rotate(surface_April, angle=-45,reshape='True')#,mode='wrap')
    # plt.figure();plt.imshow(z,
    #            extent=ext,
    #            aspect='auto');plt.colorbar()
    








