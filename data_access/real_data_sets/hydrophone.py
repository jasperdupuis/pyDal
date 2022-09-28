# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:12:49 2022

@author: Jasper
"""
import pandas as pd
import pickle

import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures

# The list of runs I want to look at. Later, this will be an interable over
# the dataframe's runIDs
list_run_IDs = ['DRJ1PB05AX00WB','DRJ1PB13AX00WB'] # For testing 

# the dataframe that holds runIDs and filenames
trial_runs_file = 'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'
trial_binary_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'
trial_track_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\TRACKING\\'
df = pd.read_csv(trial_runs_file)

# range processing information lives here. 
range_dictionary = signatures.data.range_info.dynamic_patbay_2019.RANGE_DICTIONARY


all_time_series = dict()
for runID in list_run_IDs:
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
    temp['X'] = df_temp[ range_dictionary['Propeller X string'] ]
    temp['Y'] = df_temp[ range_dictionary['Propeller Y string'] ]
    temp['Time'] = df_temp[ range_dictionary['Time string'] ]



    all_time_series[runID] = temp
    
fname = r'data_timeseries.pkl'
with open(fname, 'wb') as f:
    pickle.dump(all_time_series, f)
