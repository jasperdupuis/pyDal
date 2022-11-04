# -*- coding: utf-8 -*-
"""
Script to use the functions across this module


@author: Jasper
"""

import h5py as h5
import numpy as np
import os

import matplotlib.pyplot as plt

import data_access.real_data_sets.hydrophone as hydrophone
import data_access.real_data_sets.ambients as ambients
from data_access.real_data_sets.accessor_class import RealDataManager as mgr_real


import data_analysis.statistics as mystats

# These are for 0.1 s windows
# INDEX_FREQ_LOW = 1
# INDEX_FREQ_HIGH = 8999 #90k cutoff

# These are for  1.0s windows
INDEX_FREQ_LOW = 3
INDEX_FREQ_HIGH = 89999 #90k cutoff


TARGET_FREQ = 1000 # The freq for analysis
speed='07'        
DAY = 'DRJ3'


if __name__ == '__main__':
    # data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_10/'
    data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_01_overlap_90/'
    mgr = mgr_real()
    list_files = os.listdir(data_dir)
    list_runs = [x.split('_')[0] for x in list_files]


    # Get the freq basis
    freq_basis_trimmed = ambients.get_freq_basis(list_runs[0],
                                                p_index_low  =  INDEX_FREQ_LOW,
                                                p_index_high = INDEX_FREQ_HIGH,
                                                p_data_dir = data_dir)
    
    # Generate the calibrations if used later.
    cal_s, cal_n = hydrophone.get_and_interpolate_calibrations(freq_basis_trimmed,
                                                               p_target_bw = 1)
    
    # Get all the ambients, freqs_ret is the freqs from all the STFT 
    # ( for checking if wanted. )
    # FOR NOW, set cal_S and cal_n to be zero here to easily compare
    amb_s,amb_n,runs_used,freqs_ret = ambients.return_ambient_results( ambients.GOOD, #Not all ambient runs are good.
                                                                      np.zeros_like(cal_n),
                                                                      np.zeros_like(cal_s),
                                                                      INDEX_FREQ_LOW,
                                                                      INDEX_FREQ_HIGH,
                                                                      data_dir)
    
    #
    # find the desired freq's index within the freq basis 
    target_freq = TARGET_FREQ
    # targ = 200
    # target_freq = targ
    target_index = np.where(freq_basis_trimmed  - target_freq > 0)[0][0] # get the first value
    target_index = target_index - 1
    
    
    # The below plots the spectral time series for a selected frequency.        
    plt.figure()     
    series = dict()
    # Adds the run data:
    for runID in list_runs:
        if (DAY not in runID): continue # Day selection
        if (speed not in runID): continue
        if 'frequency' in runID: continue # don't want this run.
        fname = data_dir + '\\' + runID + r'_data_timeseries.hdf5'           
        with h5.File(fname, 'r') as file:
            try:
                spec_n = file['North_Spectrogram'][:]
                t_n = file['North_Spectrogram_Time'][:]
                spec_s = file['South_Spectrogram'][:]
                t_s = file['South_Spectrogram_Time'][:]
                x = file['X'][:]
                y = file['Y'][:]
                r = np.sqrt(x*x + y*y) 
                index_cpa = np.where(r==np.min(r))[0][0]
            except:
                print (runID + ' didn\'t work')
                continue
            # Keep what follows at this loop level due to continue logic above.
            # North hydrophone HARD CODED MAGIC VARIABLES:
            samp_n = 10*np.log10(spec_n[target_index,:])
            t_nn = t_n-np.min(t_n)
            # plt.plot(t_nn,samp_n[::-1],marker='.',linestyle='None',label=runID) 
            # plt.axvline(t_nn[index_cpa])
            # DO flip the westbound runs for more real comparison to eastbound.
            if ('W' in runID): 
                samp_n = samp_n[::-1]
                plt.plot(t_nn,samp_n,marker='.',linestyle='None',label=runID) 
                plt.axvline(t_nn[ len( t_nn ) - index_cpa ] )
            else: 
                plt.plot(t_nn,samp_n,marker='.',linestyle='None',label=runID)
                plt.axvline( t_nn [ index_cpa ] )
            series[runID] = samp_n  #for later cross correlation

    # Adds the ambient data as horizontal lines.
    # Convert to arrays
    amb_nn = np.array(amb_n)
    select = amb_nn[:,target_freq]
    for r,s in zip(runs_used,select):
        if r[:4] == 'AMJ1': plt.axhline(s,color='c')
        if r[:4] == 'AMJ2': plt.axhline(s,color='b')
        if r[:4] == 'AMJ3': plt.axhline(s,color='r')
    plt.title(str(target_freq) + ' Hz with ambient received levels as horizontal lines \n 1 Hz BW, db ref V^2')
    plt.legend()
    plt.show()        












