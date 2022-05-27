# -*- coding: utf-8 -*-
"""

Get time series data, trimmed and everything, ready for further uniform processing

Data:
Hydrophones - Orca 2020 runs at 9 and 11 knots 
Accels - HDW and MAR 2021 at 7 knots

"""

DUMP_DATA = False

import pandas as pd
import numpy as np
import nptdms
import pickle

#import pydrdc
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures

trial_runs_dir = r'C:\Users\Jasper\Desktop\MASC\DSP\data_dsp\\'
trial_runs_file = trial_runs_dir + 'DSP_file_map.csv'
df = pd.read_csv(trial_runs_file)


keys = dict()
keys['9 knot hydrophone measurement'] = 'DRF2PB09AA00EB' # len(str) = 14
keys['11 knot hydrophone measurement'] = 'DRF2PB11AA00EB' # len(str) = 14
keys['7 knot accelerometer measurement June'] = '070HF00AES'
keys['7 knot accelerometer measurement Dec (1)'] = '070ST00AEX'
keys['7 knot accelerometer measurement Dec (2)'] = '070ST20AEX'


def divide_time_series(p_data,
                       p_overlap = 0, # in decimal 0 to 1.
                       p_fs = 25600,
                       p_seconds = 3):
    len_win = int(p_seconds*p_fs)
    shift = len_win*(1-p_overlap)
    last_effective_index = len(p_data) - len_win
    N_win = int(last_effective_index//shift)
    result=np.zeros([len_win,N_win])
    time_stamps = []
    for index in range(N_win):
        start = int(index*shift)
        end = int(start + len_win)
        result[:,index] = p_data[start:end]
        time_stamps.append(start/p_fs)
    time_stamps = np.array(time_stamps)
    return result, time_stamps

def load_data_spec():
    with open (trial_runs_dir + r'\\results\\spectrograms_avg.txt', 'rb') as fp:
        spec_dictionary = pickle.load(fp)    
    keys = list(spec_dictionary.keys())
    return keys,spec_dictionary

def load_data_time():
    with open (trial_runs_dir + r'\\time_series\\time_series.txt', 'rb') as fp:
        time_series_dictionary = pickle.load(fp)    
    keys = list(time_series_dictionary.keys())
    return keys,time_series_dictionary        
    
def create_data(DUMP_DATA=True):
    #range data requirements
    range_dictionary = signatures.data.range_info.dynamic_patbay_2020.RANGE_DICTIONARY
    
    #accel data requirements
    LocationDesc = 'PMR mid lower' #MAR location and key info in var name 
    DeckAndCompartment = 'MidPortMotorBottom' #HDW location and key info in var name

    time_series = dict()    
    for key,value in keys.items():
        row = df[df['Run ID'] == value]
        if len(value) > 12: #it's hydrophone data
            fname = trial_runs_dir + row['Hydrophone file'].values[0]
            hyd = \
                signatures.data.range_hydrophone.Range_Hydrophone_Canada(range_dictionary)
            hyd.load_range_specifications(range_dictionary)
            uncalibratedDataFloats, labelFinder, message = hyd.load_data_raw_single_hydrophone(fname)
            time_series[key] = uncalibratedDataFloats
            
        else: #it's accel data
            fname = trial_runs_dir + row['PMR TDMS File'].values[0]
            td = nptdms.TdmsFile.read(fname)
            channels = td.groups()[0]._channels
            for gibberish,c in channels.items():
                if value[-1] == 'X': #it's December data
                    if c.properties['LocationDesc'] == LocationDesc:
                        time_series[key] = c.raw_data
                        break
            
                if value[-1] == 'S': #it's June data
                    if c.properties['DeckAndCompartment'] == DeckAndCompartment:
                        time_series[key] = c.raw_data
                        break

    if DUMP_DATA:         
        with open(trial_runs_dir + r'\\time_series\\time_series.txt', 'wb') as fp:
            pickle.dump(time_series, fp)                  
