# -*- coding: utf-8 -*-
"""
Script to use the functions across this module


@author: Jasper
"""

import h5py as h5
import numpy as np
import pandas as pd
import os
import pickle 


import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


import data_access.real_hydrophone as hydrophone
import data_access.real_ambients as ambients
from data_access.real_accessor_class import Real_Data_Manager as mgr_real
from env.bathymetry import Bathymetry_CHS


TARGET_FREQ = 73 # The freq for analysis
NUM_DAY = '3'

TYPE = 'DR'
MTH = 'J'
STATE = 'A'
SPEED='07'        
HEADING = 'X' #X means both

DAY = TYPE + MTH + NUM_DAY

# Data directory of interest, note there are a few different ways of making spectrograms
dir_specgram = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_01_overlap_90/'
# Contains kurtosis, tracks, etc.
summary_fname = r'summary_stats_dict.pkl'


# Maps config-speed pairs to lists of runIDs matching those parameters. 
# DAY AGNOSTIC
fname_config_trial_map = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal/data_access/config_speed_run_dictionary.pkl'

fname_speed_rpm = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal/data_access/real_data_sets/RPM_shaft_map.txt'


if __name__ == '__main__':
    # data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_10/'
    list_files = os.listdir(dir_specgram)
    list_runs = [x.split('_')[0] for x in list_files]
    TEST_RUN = list_runs[40]
    mgr = mgr_real(list_runs,dir_specgram)    

    summary_stats = mgr.load_and_set_pickle_summary_stats(
        dir_specgram + summary_fname)
    config_dictionary = mgr.get_config_dictionary(fname_config_trial_map)
    
    bathy = Bathymetry_CHS()
    bathy.get_2d_bathymetry_trimmed( #og variable values
                                  p_location_as_string = 'Patricia Bay',
                                  p_num_points_lon = 200,
                                  p_num_points_lat = 50,
                                  p_lat_delta = 0.00095,
                                  p_lon_delta = 0.0015,
                                  p_depth_offset = 0)
    fig_bathy,ax_bathy = bathy.plot_bathy()
    
    """
    #
    Get selected runs based on MACRO_PARAMS above, and then get their xy
    from the hdf5 timeseries file.
    Timeries in particular chosen by the directory taken from.
    #
    """

    runs = mgr.get_run_selection(
                                p_type = TYPE,
                                p_mth = MTH,
                                p_machine = STATE,
                                p_speed = SPEED,
                                p_head = HEADING)
    
    tracks = mgr.get_track_data(runs,dir_specgram)
    
    for key,value in tracks.items():
        X = value['Y']
        Y = value['X']
        ax_bathy.scatter(X,Y,marker='.',label=key)
    plt.legend()


    # A scatter plot of multiple runs with given processing (data dir sets this)
    mgr.scatter_selected_data_single_f(TARGET_FREQ,
                                        DAY,
                                        SPEED)     

    mgr.set_rpm_table(fname_speed_rpm)
    freq_shaft = mgr.return_nominal_rpm_as_hz(SPEED)