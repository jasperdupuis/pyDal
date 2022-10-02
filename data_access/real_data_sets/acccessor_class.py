# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:02:21 2022

@author: Jasper
"""

import numpy as np
import h5py as h5

import hydrophone


class DataAccessor():
    
    def __init__(self,
                 p_freq,
                 p_runs,
                 p_data_directory = r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal\data_access\real_data_sets\hdf5_timeseries/'):
        self.runs = p_runs
        self.data_dir = p_data_directory
        self.set_freq(p_freq)
        

        return self

    def get_data_set(self):
        for runID in self.runs:
            fname = self.data_dir' + runID + r'_data_timeseries.hdf5'           
            if not(os.path.exists(fname)): continue
            temp = dict()
            row = df[ df ['Run ID'] == runID ]
            for runID in self.runs:
            

    
  
    def get_calibrations(self,p_target_freq_basis):
        s,n = hydrophone.get_and_interpolate_calibrations(p_target_freq_basis)
        self.south_cal = s
        self.north_cal = n

    def set_freq(self,p_freq):
        self.freq = p_freq














        