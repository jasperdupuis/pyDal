# -*- coding: utf-8 -*-
"""

Build some test signals for testing.

Could be ship time series, modulation schemes, etc

"""

import scipy
import numpy as np
import matplotlib.pyplot as plt


def bits_from_time_and_bitrate(p_t,
                               p_bit_rate,
                               p_cutoff = 0.5):
    """
    Generate a random bitstream that maps the passed bitrate on to the
    passed timebase
    
    Does not check to see if they are factors of one another
    - this is a requirement.
    (len(p_t)//bitrate 0 only)
     
    p_cutoff sets the ratio between 1 and 0 in the digital data.
    """
    # how many total bits are to be included in the passed timebase
    num_bits = int(np.floor(max(p_t) * p_bit_rate))
    bits_time = np.random.rand(num_bits)  # random numbers
    bits_time = (bits_time > p_cutoff).astype(int) # the bits as 1 and 0
    # this is the number of entries in the time base for a given bit
    bit_length = len(p_t)//num_bits 
    bit_basis = np.ones(bit_length)
    result = np.zeros_like(p_t)
    for index in range(len(bits_time)):
        start = int(index * bit_length)
        end = int(( index + 1 ) * bit_length)
        result[start: end] = bits_time[index] * bit_basis
    return result
    

def random_noise(p_t):
    n_t = np.random.rand(len(p_t))
    return n_t

def random_BPSK(p_t,
                p_fc=1e4,
                p_bit_rate=100,
                p_noise_power=0.1 #This is -10dB from unity of signal
                ):

    bit_basis = bits_from_time_and_bitrate(p_t,p_bit_rate)
    arg = (2*np.pi*p_bit_rate*p_t) + (np.pi * bit_basis)
    x_t = np.cos(arg) # BPSK baseband
    c_t = np.cos(2*np.pi*p_fc*p_t) # cosine carrier  
    n_t = np.random.rand(len(x_t))
    n_t = n_t - 0.5 # remove bias for AWGN
    s_t =  (x_t * c_t) + (p_noise_power * n_t)    
    return s_t

def amplitude_modulation_pure(p_t,
                              p_fc=5e4,
                              p_fm=1e4):
    m_t = np.cos(2*np.pi*p_fm*p_t) # cosine carrier  
    c_t = np.cos(2*np.pi*p_fc*p_t) # cosine carrier  
    return m_t*c_t



"""
num_seconds = 2
fs = 1e4
t = np.arange(num_seconds*fs)/fs
fc = 1e3
bit_rate = 100
s_t = random_BPSK(t,fc,bit_rate,0.01)

# plt.plot(t,s_t)
# plt.plot(t,x)
fft = np.fft.fft(s_t)
freqs = np.fft.fftfreq(len(s_t),1/fs)
plt.plot(freqs,np.abs(fft))
"""


