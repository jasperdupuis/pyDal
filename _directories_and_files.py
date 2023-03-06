"""

A file to hold all explicit strings for directory and file management

import example:
    
from directories_and_files import DIR_SPETROGRAM, SUMMARY_FNAME

"""

"""

DIRECTORIES

"""
# # Data directory of interest, note there are a few different ways of making spectrograms
# # The 90% overlap with 1s windows ==> 1Hz resolution, 10 points per s
# DIR_SPECTROGRAM = \
#     r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_01_overlap_90\\'
# The 0 overlap and 0.1s windows ==> 10Hz resolution, 1 point per s
DIR_SPECTROGRAM = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_10\\'


DIR_BINARY_HYDROPHONE = \
    r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'

DIR_TRACK_DATA = \
    r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\TRACKING\\'


""" 

RELATIVE FILE NAMES 

"""

# Contains kurtosis, tracks, etc.
SUMMARY_FNAME = r'summary_stats_dict.pkl'


"""

EXPLICIT FILE NAMES

"""
# Maps config-speed pairs to lists of runIDs matching those parameters. 
FNAME_CONFIG_TRIAL_MAP =\
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal/data_access/config_speed_run_dictionary.pkl'
# The shaft speed - RPM table in a space-delimited file.
FNAME_SPEED_RPM = \
    r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal/data_access/real_data_sets/RPM_shaft_map.txt'

TRIAL_MAP = \
    r'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'

FILE_NORTH_CAL = \
    r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/TF_DYN_NORTH_L_40.CSV'
FILE_SOUTH_CAL = \
    r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/TF_DYN_SOUTH_L_40.CSV'


