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

# These are for 0.1 s windows
# INDEX_FREQ_LOW = 1
# INDEX_FREQ_HIGH = 8999 #90k cutoff

# These are for  1.0s windows
INDEX_FREQ_LOW = 3
INDEX_FREQ_HIGH = 89999 #90k cutoff


ambient_data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_01_overlap_90/'

OOPS = [ #runs the range fucked up for sure:
        'AMJ1PB00XX00XX',
        'AMJ1PB00XX01XX',
        'AMJ1PB00XX02XX',
        'AMJ1PB00XX04XX',
        'AMJ1PB00XX05XX',
        'AMJ2PB00XX01XX',
        'AMJ2PB00XX02XX',
        'AMJ3PB00XX00XX']

GOOD = ['AMJ1PB00XX03XX', 
        'AMJ1PB00XX06XX',
        'AMJ1PB00XX07XX',
        'AMJ1PB00XX08XX',
        'AMJ1PB00XX09XX',
        'AMJ1PB00XX10XX',
        'AMJ1PB00XX11XX',
        'AMJ2PB00XX03XX',
        'AMJ3PB00XX01XX',
        'AMJ3PB00XX02XX']

def get_freq_basis(p_runID,
                   p_index_low,
                   p_index_high,
                   p_data_dir = ambient_data_dir):    
    """
    """
    # Gets the freq basis cut to size.
    freq_basis = 0 # Declare out of with-as file scope so it sticks around.
    fname = p_data_dir + '\\' + p_runID + r'_data_timeseries.hdf5'           
    with h5.File(fname, 'r') as file:
        try:
            freq_basis =file['Frequency'][:]
        except:
            print (list_runs[0]+ ' (listruns[0]) didn\'t work to get frequency')
    freq_basis_interp = freq_basis[p_index_low:p_index_high] # 0 is out of the interpolation range, also not interested.
    return freq_basis_interp


def return_ambient_results(p_list_runs,
                           p_cal_n,
                           p_cal_s,
                           p_index_low = 3,
                           p_index_high = 8998, #defaults are for 10hz bw
                           p_data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries'):
    """
    OOPS is a hardcoded list of known bad ambients.
    Cannot be passed, takes value from header of this file.
    
    returns results as lists, can convert to array outside.
    """
    amb_s = []
    amb_n = []
    freqs = []
    runs_used = []
    for runID in p_list_runs:
        if not (runID[:2] == 'AM') : continue # only ambients allowed!
        if runID in OOPS: continue
        if 'frequency' in runID: continue # don't want this run.
        runs_used.append(runID)
        fname = p_data_dir + '\\' + runID + r'_data_timeseries.hdf5'           
        with h5.File(fname, 'r') as file:
            try:
                spec_n = file['North_Spectrogram'][:]
                spec_s = file['South_Spectrogram'][:]
                f = file['Frequency']
            except:
                print (runID + ' didn\'t work')
            
            freqs.append(f)
            
            samp_n = np.mean(spec_n,axis=1)
            samp_n = 10*np.log10(samp_n)
            samp_n = samp_n[p_index_low:p_index_high] + p_cal_n
            amb_n.append(samp_n)
            
            samp_s = np.mean(spec_s,axis=1)
            samp_s = 10*np.log10(samp_s)
            samp_s = samp_s[p_index_low:p_index_high] + p_cal_s
            amb_s.append(samp_s)
            
    return amb_s,amb_n,runs_used,freqs


if __name__ == '__main__':
    list_files = os.listdir(ambient_data_dir)
    list_runs = [x.split('_')[0] for x in list_files if 'AM' in x] # it's like English!
    
    # Get the freq basis
    freq_basis_trimmed = get_freq_basis(list_runs[0],
                                        INDEX_FREQ_LOW,
                                        INDEX_FREQ_HIGH)
    
    # Generate the calibrations if used later.
    import hydrophone
    cal_s, cal_n = hydrophone.get_and_interpolate_calibrations(freq_basis_trimmed)
    
    # Get all the ambients, freqs_ret is the freqs from all the STFT 
    # ( for checking if wanted. )
    amb_s,amb_n,runs_used,freqs_ret = return_ambient_results(list_runs,
                                                             cal_n,
                                                             cal_s,
                                                             INDEX_FREQ_LOW,
                                                             INDEX_FREQ_HIGH,
                                                             ambient_data_dir)
            
    # finally this looks reasonable.
    plt.figure()
    good = []
    for r,entry in zip(runs_used,amb_n):
        if r in OOPS: continue
        plt.plot(freq_basis_trimmed,entry,label=r)
        good.append(r)
    plt.xscale('log')
    plt.legend()        