# -*- coding: utf-8 -*-
"""
Script to use the functions across this module


@author: Jasper
"""

import h5py as h5
import numpy as np
import os
import pickle 


import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


import data_access.real_hydrophone as hydrophone
import data_access.real_ambients as ambients
from data_access.real_accessor_class import Real_Data_Manager as mgr_real
from env.bathymetry import Bathymetry_CHS

TARGET_FREQ = 16 # The freq for analysis
NUM_DAY = '3'

SPEED='05'        
TYPE = 'DR'
MTH = 'J'
STATE = 'A'
SPEED = '05'
HEADING = 'X' #X means both

DAY = TYPE + MTH + NUM_DAY


if __name__ == '__main__':
    # data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_10/'
    data_dir = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_01_overlap_90/'
    list_files = os.listdir(data_dir)
    list_runs = [x.split('_')[0] for x in list_files]
    mgr = mgr_real(list_runs,data_dir)    

    # A scatter plot of multiple runs with given processing (data dir sets this)
    # mgr.scatter_selected_data_single_f(TARGET_FREQ,
    #                                     DAY,
    #                                     SPEED) 


    # keys = list(summary_stats.keys())
    # k = keys[20]
    # print(k)
    # r = summary_stats[k]
    # plt.figure();plt.plot(mgr.freq_basis_trimmed,r['North_Skew']);plt.xscale('log');plt.title('Skew')
    # plt.figure();plt.plot(mgr.freq_basis_trimmed,r['North_Kurtosis']);plt.xscale('log');plt.title('Kurtosis')
    # plt.figure();plt.plot(mgr.freq_basis_trimmed,r['North_Scintillation_Index']);plt.xscale('log');plt.title('SI')
    
    # # RPMs: cutoff is 8 knots, 8 knots is in the high group.
    # speed_nominal = float(k[6:8])
    # shaft_hz_low = 27.676 * speed_nominal / 60
    # shaft_hz_high = 31.699 * speed_nominal / 60
    
    
    # TARGET_FREQ = 3800 # placeholder
    # mgr.scatter_selected_data_single_f(TARGET_FREQ,
    #                                     k[:4],
    #                                     speed_nominal) 


"""
#
 Prints the number of available runs for a given config speed pair.
#
"""
# fname = r'config_speed_run_dictionary.pkl'
# with open(fname, 'rb') as file:
#     config_run_dictionary= pickle.load(file)   

# keys = list(config_run_dictionary.keys())  

# for k in keys:
#     print(k + ' ' +str(len(config_run_dictionary[k])))
# config_run_dictionary[keys[15]]



bathy = Bathymetry_CHS()
bathy.get_2d_bathymetry_trimmed( #og variable values
                              p_location_as_string = 'Patricia Bay',
                              p_num_points_lon = 200,
                              p_num_points_lat = 50,
                              p_lat_delta = 0.00095,
                              p_lon_delta = 0.0015,
                              p_depth_offset = 0)

# getting the original colormap using cm.get_cmap() function
orig_map=plt.cm.get_cmap('viridis')
# reversing the original colormap using reversed() function
reversed_map = orig_map.reversed()
plt.figure();plt.imshow(bathy.z_interped,
                        extent = bathy.ext,
                        cmap = reversed_map,
                        origin = 'upper',
                        aspect = 'auto');plt.colorbar()

"""
#
Get selected runs based on MACRO_PARAMS above, and then get their xy
from the hdf5 timeseries file.
Timeries in particular chosen by the directory taken from.
#
"""

dir_tseries = \
    r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries_bw_01_overlap_90'

runs = mgr.get_run_selection(
                            p_type = TYPE,
                            p_mth = MTH,
                            p_machine = STATE,
                            p_speed = SPEED,
                            p_head = HEADING)

tracks = mgr.get_track_data(runs,dir_tseries)

for key,value in tracks.items():
    X = value['Y']
    Y = value['X']
    plt.scatter(X,Y,marker='.',label=key)
plt.legend()








