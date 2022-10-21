# -*- coding: utf-8 -*-
"""

Having already accessed the ambient .bin files to make spectrograms,
look at them in various ways and perhaps store.

Created on Thu Oct 13 15:38:34 2022

@author: Jasper
"""

import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt

import hydrophone

INDEX_FREQ_10 = 1
INDEX_FREQ_90000 = 8999
# TARGET_FREQ = 100 # The freq for analysis

data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries'
list_files = os.listdir(data_dir)
list_runs = [x.split('_')[0] for x in list_files if 'AM' in x] # it's like English!

OOPS = [ #runs the range fucked up for sure:
        'AMJ2PB00XX01XX',
        'AMJ1PB00XX04XX',
        'AMJ1PB00XX05XX']

# Get the freq basis
freq_basis = 0 # Declare out of with-as file scope so it sticks around.
freq_name = data_dir + '\\frequency_basis.hdf5'           
with h5.File(freq_name,'r') as file:    
    freq_basis = file['Frequency'][:]


# Generate the calibrations if used later.
freq_basis_interp = freq_basis[INDEX_FREQ_10:INDEX_FREQ_90000] # 0 is out of the interpolation range, also not interested.
cal_s, cal_n = hydrophone.get_and_interpolate_calibrations(freq_basis_interp)


# find the desired freq's index within the freq basis 
# target_freq = TARGET_FREQ
# target_index = np.where(freq_basis - target_freq > 0)[0][0] # get the first value
# target_index = target_index - 1


amb_s = []
amb_n = []
runs_used = []
for runID in list_runs:
    if runID in OOPS: continue
    if 'frequency' in runID: continue # don't want this run.
    runs_used.append(runID)
    fname = data_dir + '\\' + runID + r'_data_timeseries.hdf5'           
    with h5.File(fname, 'r') as file:
        try:
            spec_n = file['North_Spectrogram'][:]
            spec_s = file['South_Spectrogram'][:]
        except:
            print (runID + ' didn\'t work')
        
        samp_n = np.mean(spec_n,axis=1)
        samp_n = 10*np.log10(samp_n)
        samp_n = samp_n[INDEX_FREQ_10:INDEX_FREQ_90000] + cal_n
        amb_n.append(samp_n)
        
        samp_s = np.mean(spec_s,axis=1)
        samp_s = 10*np.log10(samp_s)
        samp_s = samp_s[INDEX_FREQ_10:INDEX_FREQ_90000] + cal_s
        amb_s.append(samp_s)
   
x = samp_s    
xf = freq_basis_interp
xdf = df[df.columns[1]].values
fdf = df[df.columns[0]].values

plt.figure()
plt.plot(xf,x+cal_s,label='My calc')
plt.plot(fdf,xdf,label='Range OTO')
plt.legend()
    
plt.figure()
for r,entry in zip(runs_used,amb_n):
    plt.plot(freq_basis_interp,entry,label=r)
plt.xscale('log')
plt.legend()        