# -*- coding: utf-8 -*-
"""
Script to use the functions across this module


@author: Jasper
"""

import sys
sys.path.append('C:\\Users\\Jasper\\Documents\\Repo\\pyDal\\pyDal\\env')

TARGET_FREQ = 101 # The freq for analysis
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
    from scripts import analysis_main
    # from env import create_TL_models
    
    
    