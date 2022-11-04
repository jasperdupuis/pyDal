# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:02:21 2022

@author: Jasper
"""

import numpy as np
import h5py as h5

from . import hydrophone
from . import ambients



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
    
    def __init__(self):
        
        self.FS_HYD = 204800
        self.T_HYD = 1.5 #window length in seconds
        self.FS_GPS = 10
        self.LABEL_COM = 'COM '
        self.LABEL_FIN = 'FIN '


    def load_target_spectrogram_data(self,p_runID,p_data_dir):
       """
       given a runID and data directory load the results to a dictionary.
       
       Remember, the arrays accessed here are in linear values.
       """    
       result = dict()
       fname = p_data_dir + '\\' + p_runID + r'_data_timeseries.hdf5'           
       with h5.File(fname, 'r') as file:
           try:
               result['North_spectrogram'] = file['North_Spectrogram'][:]
               result['North_spectrogram_Time']= file['North_Spectrogram_Time'][:]
               result['South_spectrogram']= file['South_Spectrogram'][:]
               result['South_spectrogram_Time']= file['South_Spectrogram_Time'][:]
               result['X'] = file['X'][:]
               result['Y'] = file['Y'][:]
               result['R'] = np.sqrt(result['X']*result['X'] + result['Y']*result['Y']) 
               result['CPA_Index'] = np.where(result['R']==np.min(result['R']))[0][0]
           except:
               print (p_runID + ' didn\'t work')
               print (p_runID + ' will have null entries')
       return result
  
    def get_calibrations(self,p_target_freq_basis):
        s,n = hydrophone.get_and_interpolate_calibrations(p_target_freq_basis)
        self.south_cal = s
        self.north_cal = n


    def set_freq(self,p_freq):
        """
        Only a single freq of interest gets used at a time.
        """
        self.freq = p_freq


   












        