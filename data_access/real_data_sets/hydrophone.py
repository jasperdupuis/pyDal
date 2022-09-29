# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:12:49 2022

@author: Jasper
"""

import pandas as pd
from scipy import interpolate
import h5py as h5
import os

import numpy as np

import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures

FS_HYD = 204800
T_HYD = 1.5 #window length in seconds
FS_GPS = 10
LABEL_COM = 'COM '
LABEL_FIN = 'FIN '

# The list of runs I want to look at. Later, this will be an interable over
# the dataframe's runIDs
# list_run_IDs = ['DRJ1PB05AX00WB','DRJ1PB13AX00WB'] # For testing 

# the dataframe that holds runIDs and filenames
trial_runs_file = 'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'
trial_binary_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'
trial_track_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\TRACKING\\'
df = pd.read_csv(trial_runs_file)

list_run_IDs = df[ df.columns[1] ].values

# range processing information lives here. 
range_dictionary = signatures.data.range_info.dynamic_patbay_2019.RANGE_DICTIONARY

def align_track_and_hyd_data(p_the_run_dictionary,
                             labelFinder,
                             label_com = LABEL_COM,
                             label_fin = LABEL_FIN,
                             fs_hyd = FS_HYD,
                             t_hyd = T_HYD):
    """
    ASSUMPTION: len(hydrophone data time) >= len (gps data time)
    That is, we only need to prune the hydrophone data to match gps data.
    So then in that case there are four cases:
        1) Missing both labels
        2) Missing COM
        3) Missing FIN
        4) Both labels present
    Each case must be treated.
    Further there is the potential that len(hyd_data) / fs > len(gps)/10
    even after truncation!
    In this case find the delta and split it evenly between start and end.    
    
    input dictionary depends on having keys North, South, Time
    
    labelFinder is the list returned from pydrdc.signature.loadHydData.
    """
    # STEP ONE: Get the labels indices or set them to 0,-1
    try:
        index_com = labelFinder.index(label_com)
    except:
        index_com = 0
    try:
        index_fin = labelFinder.index(label_fin)    
    except:
        index_fin = -1
    # STEP TWO: Apply the label indices to the hydrophone data.
    if index_fin == -1: # Do not want to multiply -1 by fs.
        start = int(index_com * fs_hyd * t_hyd)
        end = int(index_fin)
        p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ start : end ]
        p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ start : end ]
    else: # index IS meaningful, so use it.
        start = int(index_com * fs_hyd * t_hyd)
        end = int(index_fin * fs_hyd * t_hyd)
        p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ start : end ]
        p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ start : end ]
    # STEP THREE: Check if signal lengths are good:
    time_g = p_the_run_dictionary['Time'][-1] - p_the_run_dictionary['Time'][0] # Use this in case samples are missed.
        # Treat time_g for float rounding - only want the first decimal place
    time_g = int(time_g * 10) / 10
    time_h = len(temp['North'])/fs_hyd
    if not(time_g == time_h):
        #So, the total hydrophone time is not equal to the total gps time elapsed
        dt = time_h - time_g
        dt = np.round(dt,2)
        trunc_one_ended = int(fs_hyd * dt/2) # amount of data to chop from each end
        p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ trunc_one_ended : -1 * trunc_one_ended ]
        p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ trunc_one_ended : -1 * trunc_one_ended ]
    else:
        # The unlikely case of  gps and hyd times aligning.
        # null operation required
        p_the_run_dictionary['North'] = p_the_run_dictionary['North']        
        p_the_run_dictionary['South'] = p_the_run_dictionary['South']

    return p_the_run_dictionary

def interpolate_x_y(p_the_run_dictionary):
    # Now, must make sure there is an x,y sample for each time step.
    # Note ther eare missing time steps but we know they occured, so 
    # interpolate away!
    # 2x 1d interpolations for each of x, y
    x_function = interpolate.interp1d(
        p_the_run_dictionary['Time'], # x
        p_the_run_dictionary['X'])    # f(x)
    y_function = interpolate.interp1d(
        p_the_run_dictionary['Time'], # x
        p_the_run_dictionary['Y'])    # f(x)
    t = np.arange(
        p_the_run_dictionary['Time'][0], #start
        p_the_run_dictionary['Time'][-1], #stop
        1/FS_GPS)                        #step
    p_the_run_dictionary['X'] = x_function(t)
    p_the_run_dictionary['Y'] = y_function(t)
    p_the_run_dictionary['Time'] = t
    return p_the_run_dictionary

for runID in list_run_IDs:
    if 'DRJ' not in runID: continue #only want 2019 dynamic data.
    fname = 'hdf5_timeseries/' + runID + r'_data_timeseries.hdf5'           
    if not(os.path.exists(fname)): continue
    temp = dict()
    row = df[ df ['Run ID'] == runID ]
    
    fname = trial_binary_dir + row['South hydrophone raw'].values[0]
    hyd = \
        signatures.data.range_hydrophone.Range_Hydrophone_Canada(range_dictionary)
    hyd.load_range_specifications(range_dictionary)
    uncalibratedDataFloats, labelFinder, message = hyd.load_data_raw_single_hydrophone(fname)
    temp['South'] = uncalibratedDataFloats
    
    fname = trial_binary_dir + row['North hydrophone raw'].values[0]
    hyd = \
        signatures.data.range_hydrophone.Range_Hydrophone_Canada(range_dictionary)
    hyd.load_range_specifications(range_dictionary)
    uncalibratedDataFloats, labelFinder, message = hyd.load_data_raw_single_hydrophone(fname)
    temp['North'] = uncalibratedDataFloats    
    
    fname = trial_track_dir + row['Tracking file'].values[0]
    track = signatures.data.range_track.Range_Track()
    track.load_process_specifications(range_dictionary)
    track.load_data_track(fname)
    start_s_since_midnight, total_s = \
        track.trim_track_data(r = range_dictionary['Track Length (m)'] / 2,
            prop_x_string = range_dictionary['Propeller X string'],
            prop_y_string = range_dictionary['Propeller Y string'],
            CPA_X = range_dictionary['CPA X (m)'],
            CPA_Y = range_dictionary['CPA Y (m)'])
    df_temp = track.data_track_df_trimmed
        
    temp['X'] = df_temp[ range_dictionary['Propeller X string'] ].values
    temp['Y'] = df_temp[ range_dictionary['Propeller Y string'] ].values
    temp['Time'] = df_temp[ range_dictionary['Time string'] ].values

    temp = align_track_and_hyd_data(temp, labelFinder) # do some truncation
    temp = interpolate_x_y(temp) # make sure the entire time base is represented

    
    try:
        os.remove(fname)
    except:
        print(runID + ' hdf5 file did not exist before generation')
        
    with h5.File(fname, 'w') as file:
        for data_type,data in temp.items():
            # note that not all variable types are supported but string and int are
            file[data_type] = data


# h = h5.File(fname)
# h.keys()
# x = h['X'][:]
# y = h['Y'][:]
# r = np.sqrt(x**2 + y**2)
# t = h['Time'][:]
# n = h['North'][:]


# len(n)/FS_HYD
# len(t)/10
# t[-1] - t[0]

# h.close()

# h




