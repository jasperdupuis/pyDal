# -*- coding: utf-8 -*-
"""

Analysis code, for parameters on lines ~ 20 - 30,
generate a time-series plot of the frequency,
and the tracks over the bathymetry.

"""

import h5py as h5
import numpy as np
import pandas as pd
import os
import pickle 


import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

import _variables

import _directories_and_files

from _imports import \
    Real_Hyd,\
    Real_Amb,\
    Real_Data_Manager,\
    Bathymetry_CHS_2,\
    Location
    


the_location= Location(_variables.LOCATION)


list_files = os.listdir(_directories_and_files.DIR_SPECTROGRAM)
list_runs = [x.split('_')[0] for x in list_files]
mgr = Real_Data_Manager(list_runs,_directories_and_files.DIR_SPECTROGRAM)    


summary_stats = \
    mgr.load_and_set_pickle_summary_stats(
    _directories_and_files.DIR_SPECTROGRAM + _directories_and_files.SUMMARY_FNAME)

config_dictionary = \
    mgr.get_config_dictionary(
        _directories_and_files.FNAME_CONFIG_TRIAL_MAP)


# bathy = Bathymetry_CHS_2()
# bathy.get_2d_bathymetry_trimmed( #og variable values
#                               p_location_as_object = the_location,
#                               p_num_points_lon = 200,
#                               p_num_points_lat = 50,
#                               # p_lat_delta = 0.00095,
#                               # p_lon_delta = 0.0015,
#                               p_depth_offset = 0)


"""
#
Get selected runs based on MACRO_PARAMS in variables.py, and then get their xy
from the hdf5 timeseries file.
Timeseries in particular chosen by the directory taken from.

Ambient doesnt show bathy/track.
#
"""

runs = mgr.get_run_selection(
                            p_type = _variables.TYPE,
                            p_mth = _variables.MTH,
                            p_machine = _variables.STATE,
                            p_speed = _variables.SPEED,
                            p_head = _variables.HEADING)

# if not(variables.TYPE == 'AM'):
#     # Ambient runs don't have track data.
#     fig_bathy,ax_bathy = \
#         bathy.plot_bathy(the_location,
#                          p_N_LAT_PTS=48,
#                          p_N_LON_PTS=198,
#                          )
#     tracks = mgr.get_track_data(runs,variables.DIR_SPECTROGRAM)
#     for key,value in tracks.items():
#         X = value['Y']
#         Y = value['X']
#         ax_bathy.scatter(X,Y,marker='.',label=key)
#     plt.legend()


# A scatter plot of multiple runs with given processing (data dir sets this)
fix,ax = mgr.scatter_selected_data_single_f(runs,
                                   _variables.TYPE,
                                   _variables.TARGET_FREQ,
                                    _variables.DAY,
                                    _variables.SPEED,
                                    _variables.HYDROPHONE,
                                    p_decibel_bool = True,
                                    p_ambients_bool = False) #Which hydrophone in use.

mgr.set_rpm_table(_directories_and_files.FNAME_SPEED_RPM)
freq_shaft = mgr.return_nominal_rpm_as_hz(_variables.SPEED)





