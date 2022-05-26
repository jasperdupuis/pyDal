# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:27:40 2022

@author: Jasper
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../../csp')
import cyclic_modulation_coherence
import data_methods
import util_plotting

# keys[3] == 7 knot accel Dec (day 1) #
keys,time_dict = data_methods.load_data_time()
keys,spec_dict = data_methods.load_data_spec()
data = time_dict[keys[3]]

# my variable names
###
# FIXED :
FS = 25600
num_seconds_chunk = 3
num_seconds_total = 60
N_chunk = FS * num_seconds_total
N_total = FS * num_seconds_total #largest effect on del_alpha and max alpha
overlap_major = 0.75 # Overlap of chunks to make a single gram
overlap_minor = 0.75 # Overlap of windows used in making a single gram, often denoted L in integer form
del_f = 25 # a major determinant in how much cyclic resolution you get. Higher delf ==> more alpha.
# SET THE DATA SET BASED ON N:
x = data[:N_total]
###

n_secs = [3,5,7,10,15,20]
r_dictionary= dict()
x = data[:N_total]
for num_seconds_chunk in n_secs:
    # reset a few required values:
    temp_dict = dict()

    N_prime,L,M,del_alpha_achieved,n_freqs,n_alphas,w = \
        cyclic_modulation_coherence.calculate_parameters(
            N = N_chunk,
            fs = FS,
            del_f = del_f,
            overlap_minor = overlap_minor)

    samples,timestamps = data_methods.divide_time_series(x,overlap_major,FS,num_seconds_chunk)
    t,freqs,alphas,r_cmc,r_spec = cyclic_modulation_coherence.create_sampled_CMC_and_gram(
                                                            samples,
                                                             fs = FS,
                                                             M = M,
                                                             w = w,
                                                             N_prime = N_prime,
                                                             L = L,
                                                             n_alphas = n_alphas,
                                                             n_freqs = n_freqs)
    temp_dict['Time (s)'] = t
    temp_dict['Spectrogram frequency (Hz)'] = freqs
    temp_dict['Cyclic frequency (Hz)'] = alphas
    temp_dict['Cyclic Modulation Coherence'] = r_cmc
    temp_dict['Spectrogram'] = r_spec
    r_dictionary[num_seconds_total] = temp_dict
    

# plotting images for comparison
nrow = 2
ncol = 3
selectors = []
for row in range(nrow):
    for col in range(ncol):
       selectors.append((row,col)) 
f,ax_arr = plt.subplots(nrow,ncol)
index = 0
data_label = 'Cyclic Modulation Coherence'
total_label =  data_label
x_label = 'Spectrogram frequency (Hz)'
y_label = 'Cyclic frequency (Hz)'
units = '10log_10(), ref arbitrary'
for index in range(len(n_secs)):
    n = n_secs[index]
    label = str(r_dictionary[n][data_label].shape[0]) + ' windows ('+str(n) + ' s)'
    _,im = util_plotting.plot_and_return_2D_axis(
        ax_arr[selectors[index]],
        np.mean(r_dictionary[n][data_label],axis=0), #3 is CMC, 4 is PSD
        r_dictionary[n][x_label],
        r_dictionary[n][y_label],
        p_x_min = 0,
        p_x_max = r_dictionary[n][x_label][-1],
        p_y_min = 0,
        p_y_max = r_dictionary[n][y_label].shape[0],
        p_label = label)
    index+= 1
# f.supylabel('Spectrogram time (s)')     #FUTURE RELEASE OF MATPLOTLIB
# f.supxlabel('Spectrogram frequency (Hz)') # FUTURE RELEASE OF MATPLOTLIB
f.text(0.5, 0.04, x_label, ha='center')
f.text(0.04, 0.5, y_label, va='center', rotation='vertical')
cbar = f.colorbar(im, ax=ax_arr.ravel().tolist())
cbar.ax.set_ylabel(units)
f.suptitle(data_label + '\n' + str(TOTAL_SECONDS) + ' of data with windows of given length results averaged')
# f.show()

f,ax_arr = plt.subplots(nrow,ncol)
index = 0
data_label = 'Cyclic Modulation Coherence'
total_label = 'Integrated '+ data_label
x_label = 'Cyclic frequency (Hz)'
units = '10log_10(), ref arbitrary'
for index in range(len(n_secs)):
    n = n_secs[index]
    label = str(r_dictionary[n][data_label].shape[0]) + ' windows ('+str(n) + ' s)'
    ICMC = np.mean(r_dictionary[n][data_label],axis=0)
    ICMC = np.mean(ICMC,axis=1)
    _,im = util_plotting.plot_and_return_1D_axis(
        ax_arr[selectors[index]],
        ICMC  ,  
        r_dictionary[n][x_label],
        p_x_min = 0,
        p_x_max = r_dictionary[n][x_label][-1],
        p_label = label,
        p_linear=True)
    
    index+= 1
f.text(0.5, 0.04, x_label, ha='center')
f.text(0.04, 0.5, y_label, va='center', rotation='vertical')
f.suptitle(total_label + '\n' + str(TOTAL_SECONDS) + ' s of data with windows of given length results averaged')
# f.show()