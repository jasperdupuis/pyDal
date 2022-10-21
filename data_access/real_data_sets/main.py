# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:58:55 2022

@author: Jasper
"""

import h5py as h5
import numpy as np
import os
import matplotlib.pyplot as plt

import hydrophone

INDEX_FREQ_10 = 1
INDEX_FREQ_90000 = 8999
TARGET_FREQ = 100 # The freq for analysis

data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries'
list_files = os.listdir(data_dir)
list_runs = [x.split('_')[0] for x in list_files]


# Set the freq basis
freq_basis = 0 # Declare out of with-as file scope so it sticks around.
freq_name = data_dir + '\\frequency_basis.hdf5'           
with h5.File(freq_name,'r') as file:    
    freq_basis = file['Frequency'][:]

# Generate the calibrations if used later.
freq_basis_interp = freq_basis[INDEX_FREQ_10:INDEX_FREQ_90000] # 0 is out of the interpolation range, also not interested.
cal_s, cal_n = hydrophone.get_and_interpolate_calibrations(freq_basis_interp)


# find the desired freq's index within the freq basis 
target_freq = TARGET_FREQ
target_index = np.where(freq_basis - target_freq > 0)[0][0] # get the first value
target_index = target_index - 1

# funct = max
runID = list_runs[5]
result_s = []
result_n = []
for runID in list_runs:
    if 'frequency' in runID: continue # don't want this run.
    fname = data_dir + '\\' + runID + r'_data_timeseries.hdf5'           
    with h5.File(fname, 'r') as file:
        try:
            spec_n = file['North_Spectrogram'][:]
            spec_s = file['South_Spectrogram'][:]
        except:
            print (runID + ' didn\'t work')
        
        samp_n = 10*np.log10(spec_n[target_index,:])
        samp_n = samp_n - np.max(samp_n)
        result_n.append(samp_n)
        
        samp_s = 10*np.log10(spec_s[target_index,:])
        samp_s = samp_s - np.max(samp_s)
        result_s.append(samp_s)
        
# The below plots the spectral time series for a selected frequency.        
# plt.figure()     
# speed='07'        
# for runID in list_runs:
#     if speed not in runID: continue
#     if 'frequency' in runID: continue # don't want this run.
#     fname = data_dir + '\\' + runID + r'_data_timeseries.hdf5'           
#     with h5.File(fname, 'r') as file:
#         try:
#             spec_n = file['North_Spectrogram'][:]
#             spec_s = file['South_Spectrogram'][:]
#             time = file['Time']
#         except:
#             print (runID + ' didn\'t work')
        
#         samp_n = 10*np.log10(spec_n[target_index,:])
#         # samp_n = samp_n - np.max(samp_n)
#         if ('W' in runID): plt.plot(time-np.min(time),samp_n[::-1],marker='.',linestyle='None',label=runID) 
#         else: plt.plot(time-np.min(time),samp_n,marker='.',linestyle='None',label=runID)
#         # plt.plot(time-np.min(time),samp_n,marker='.',linestyle='None',label=runID)
# plt.title(str(TARGET_FREQ) + ' Hz')
# plt.legend()
        











        
        
        