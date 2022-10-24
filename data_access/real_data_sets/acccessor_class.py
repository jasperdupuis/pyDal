# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:02:21 2022

@author: Jasper
"""

import numpy as np
import h5py as h5

import hydrophone



class RealDataManager():
    """
    This class should be able to simplify life, e.g.:
        process raw data to hdf5 formats
        get hydrophone time series
        get hydrophone ambients
        get hydrophone spectrograms
        retreieve spectrogram data for a target frequency
        retrieve all ambient data for a target frequency
        facilitate basic regression analysis
        facilitate future translation of this data in to ML-ready data.
    """
    
    def __init__(self,
                 p_freq,
                 p_runs,
                 p_data_directory = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries/'):
        self.runs = p_runs
        self.data_dir = p_data_directory
        self.set_freq(p_freq)
        
        self.FS_HYD = 204800
        self.T_HYD = 1.5 #window length in seconds
        self.FS_GPS = 10
        self.LABEL_COM = 'COM '
        self.LABEL_FIN = 'FIN '

        return self


    def get_data_set(self):
        for runID in self.runs:
            fname = self.data_dir + runID + r'_data_timeseries.hdf5'           
            with h5.File(fname, 'r') as file:
                spec_s = file['South_Hydrophone']
                spec_n = file['North_Hydrophone']
                freq = file['']
    
  
    def get_calibrations(self,p_target_freq_basis):
        s,n = hydrophone.get_and_interpolate_calibrations(p_target_freq_basis)
        self.south_cal = s
        self.north_cal = n


    def set_freq(self,p_freq):
        self.freq = p_freq














        