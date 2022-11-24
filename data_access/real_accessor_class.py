# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:02:21 2022

@author: Jasper
"""

import numpy as np
import h5py as h5
import scipy.stats as stats

import matplotlib.pyplot as plt

from . import real_hydrophone
from . import real_ambients


class Real_Data_Manager():
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
                 p_list_runs,
                 p_data_dir):
        self.list_runs = p_list_runs
        self.data_dir = p_data_dir
        
        #Hard coded variables below, probably dont need to touch them.
        
        # These are for  1.0s windows
        self.INDEX_FREQ_LOW = 3
        self.INDEX_FREQ_HIGH = 89999 #90k cutoff
        
        
        # These are for 0.1 s windows
        # self.INDEX_FREQ_LOW = 1
        # self.INDEX_FREQ_HIGH = 8999 #90k cutoff]
            
        self.FS_HYD = 204800
        self.T_HYD = 1.5 #window length in seconds
        self.FS_GPS = 10
        self.LABEL_COM = 'COM '
        self.LABEL_FIN = 'FIN '
        
        self.set_freq_basis()
        self.load_calibrations()
        self.load_all_ambient_data()
    
    def get_run_selection(self,
                             p_type = 'DR',
                             p_mth = 'J',
                             p_machine = 'A',
                             p_speed = '05',
                             p_head = 'X'): #X means both
        """
        Get an unordered list of runs based on simple criteria and return it
        """
        result = []
        for runID in self.list_runs:
            if (p_type not in runID[:2]): continue # Type selection
            if (p_mth not in runID[2]): continue # Month (year) selection
            if (p_machine not in runID[8]): continue # Machine config selection
            if (p_speed not in runID[6:8]): continue # peed selection
            if (p_head == 'X') : # Check for heading agnostic first.
                result.append(runID)
                continue
            elif (p_head not in runID[12]): 
                continue # Heading selection
            result.append(runID)
            
        return result
           
    
    def get_track_data(self,
                       p_runs,
                       p_dir_tseries):
        """
        Get track data straight from the time series hdf5 file.
        Returns dictionary format.
        """
        result = dict()
        for run in p_runs:
            temp_res = dict()
            fname = p_dir_tseries + '\\' + run + r'_data_timeseries.hdf5'           
            with h5.File(fname, 'r') as file:
                temp_res['X'] = file['X'][:]
                temp_res['Y'] = file['Y'][:]
            result[run] = temp_res
        return result
                
    def set_freq_basis(self):
        self.freq_basis_trimmed = real_ambients.get_freq_basis(self.list_runs[0],
                                                    p_index_low  =  self.INDEX_FREQ_LOW,
                                                    p_index_high = self.INDEX_FREQ_HIGH,
                                                    p_data_dir = self.data_dir)


    def load_calibrations(self):
            s,n = real_hydrophone.get_and_interpolate_calibrations(self.freq_basis_trimmed)
            self.cal_s  = s
            self.cal_n = n


    def load_target_spectrogram_data(self,p_runID,p_data_dir):
       """
       given a runID and data directory load the results to a dictionary.
       
       Remember, the arrays accessed here are in linear values.
       """    
       result = dict()
       fname = p_data_dir + '\\' + p_runID + r'_data_timeseries.hdf5'           
       with h5.File(fname, 'r') as file:
           try:
               result['North_Spectrogram'] = file['North_Spectrogram'][:]
               result['North_Spectrogram_Time']= file['North_Spectrogram_Time'][:]
               result['South_Spectrogram']= file['South_Spectrogram'][:]
               result['South_Spectrogram_Time']= file['South_Spectrogram_Time'][:]
               if 'AM' not in p_runID :
                   result['X'] = file['X'][:]
                   result['Y'] = file['Y'][:]
                   result['R'] = np.sqrt(result['X']*result['X'] + result['Y']*result['Y']) 
                   result['CPA_Index'] = np.where(result['R']==np.min(result['R']))[0][0]
           except:
               print (p_runID + ' didn\'t work')
               print (p_runID + ' will have null entries')
       return result
   
    
    def load_all_ambient_data(self):
        """
        Get all the ambients, freqs_ret is the freqs from all the STFT 
        ( for checking if wanted. )
        FOR NOW, set cal_S and cal_n to be zero here to easily compare
        """
        amb_s,amb_n,runs_used,freqs_ret = \
            real_ambients.return_ambient_results( 
                                            real_ambients.GOOD, #Not all ambient runs are good.
                                            np.zeros_like(self.cal_n),
                                            np.zeros_like(self.cal_s),
                                            self.INDEX_FREQ_LOW,
                                            self.INDEX_FREQ_HIGH,
                                            self.data_dir)
        self.amb_s = amb_s
        self.amb_n = amb_n
        self.amb_runs = runs_used


    
    def extract_target_frequency(self,
                                 p_run_data_dict,
                                 p_target_index):
        samp_n = 10*np.log10(p_run_data_dict['North_Spectrogram'][p_target_index,:])
        samp_s = 10*np.log10(p_run_data_dict['South_Spectrogram'][p_target_index,:])
        t_n = p_run_data_dict['North_Spectrogram_Time'] \
            - np.min(p_run_data_dict['North_Spectrogram_Time'])
        t_s = p_run_data_dict['South_Spectrogram_Time'] \
            - np.min(p_run_data_dict['South_Spectrogram_Time'])
        return samp_n, t_n, samp_s, t_s 
        
    
    def scatter_time_series(self,t,x,ax,label):
        ax.plot(t,x,marker='.',linestyle='None',label=label) 
        return ax
    
    def plot_ambient_level_single_f(self,ax):
        # Adds the ambient data as horizontal lines.
        # Convert to arrays
        amb_nn = np.array(self.amb_n)
        select = amb_nn[:,self.target_freq_index]
        for r,s in zip(self.amb_runs,select):
            if r[:4] == 'AMJ1': ax.axhline(s,color='c')
            if r[:4] == 'AMJ2': ax.axhline(s,color='b')
            if r[:4] == 'AMJ3': ax.axhline(s,color='r')
        return ax
    
        


    def set_freq(self,p_freq):
        """
        Only a single freq of interest gets used at a time.
        """
        self.freq = p_freq

    
    def find_target_freq_index(self, 
                               p_f_targ):
        """
        Map an int or float real frequency to array index.
        """
        # get the first value where this is true.
        target_index = \
            np.where(self.freq_basis_trimmed  - p_f_targ > 0)[0][0] 
        target_index = target_index - 1
        self.target_freq_index = target_index
        return target_index
        
    
    def scatter_selected_data_single_f(self,
                                       p_target_freq,
                                       p_day = 'DRJ3',
                                       p_speed = '07',
                                       p_hyd = 'NORTH'):
        """
        Loops over ALL available hdf5 data looking for runs that meet passed 
        query criteria and plots a scatter at the target freq
        """
        # find the desired freq's index within the freq basis 
        target_freq = p_target_freq
        target_f_index = self.find_target_freq_index(target_freq)
        
        # The below plots the spectral time series for a selected frequency.        
        fig,ax = plt.subplots()     
        # Adds the run data:
        for runID in self.list_runs:
            if (p_day not in runID): continue # Day selection
            if (p_speed not in runID): continue # not in selection
            if 'frequency' in runID: continue # don't want this run.
            
            runData = self.load_target_spectrogram_data(runID,  # Returns linear values
                                                       self.data_dir)
            samp_n,t_n,samp_s,t_s = self.extract_target_frequency(runData,
                                                                 target_f_index)
            
            x = runData['X'][:]
            y = runData['Y'][:]
            r = np.sqrt(x*x + y*y) 
            index_cpa = np.where(r==np.min(r))[0][0]
            
            if p_hyd == 'NORTH':
                self.scatter_time_series(t_n, samp_n, ax, label=runID)
                t_n = t_n-np.min(t_n)
                plt.axvline( t_n [ index_cpa ] )
        
            if p_hyd == 'SOUTH':
                self.scatter_time_series(t_s, samp_s, ax, label=runID)
                t_s = t_s-np.min(t_s)
                plt.axvline( t_s [ index_cpa ] )
        

        self.plot_ambient_level_single_f(ax)
        plt.title(str(target_freq) + ' Hz with ambient received levels as horizontal lines \n 1 Hz BW, db ref V^2')
        plt.legend()
        plt.show()         


    def compute_summary_stats(self):
        """
        Compute mean, std, SI, S, K for each frequency bin and run.
        """
        result = dict()
        for runID in self.list_runs:
            temp = dict()
            if 'frequency' in runID: continue # don't want this run.
            runData = self.load_target_spectrogram_data(runID,  # Returns linear values
                                                       self.data_dir)
            spec_n = runData['South_Spectrogram'][self.INDEX_FREQ_LOW : self.INDEX_FREQ_HIGH , : ]
            spec_s = runData['North_Spectrogram'][self.INDEX_FREQ_LOW : self.INDEX_FREQ_HIGH , : ]
            
            
            # Compute mean SOG if dynamic run
            if 'AM' not in runID:
                x = runData['X'][:]
                y = runData['Y'][:]
                dx = np.zeros(len(x)-10)
                dy = np.zeros(len(x)-10)
                for index in range(len(dx)):
                    dx[index] = x[index + 10] - x[index]
                    dy[index] = y[index + 10] - y[index]
                r = np.sqrt( ( dx ** 2 ) + ( dy ** 2 ))
                sog = np.mean (r) #Because it's 10 samples, the number itself is SOG
                sog_std = np.std(r)
                temp['SOG_Mean_ms'] = sog     
                temp['SOG_STD_ms'] = sog_std
            
            # Compute moments of the hydrophone spectrograms
            m_n = np.mean(spec_n,axis=1)            
            std_n = np.std(spec_n,axis=1)
            s_n = stats.skew(spec_n,axis=1)
            k_n = stats.kurtosis(spec_n,axis=1)
            si_n = ( std_n ** 2 ) / (m_n ** 2)

            m_s = np.mean(spec_s,axis=1)
            std_s = np.std(spec_s,axis=1)
            s_s = stats.skew(spec_s,axis=1)
            k_s = stats.kurtosis(spec_s,axis=1)
            si_s = ( std_s ** 2 ) / (m_s ** 2)
            
            temp['North_Mean'] = m_n            
            temp['North_STD'] = std_n
            temp['North_Skew'] = s_n
            temp['North_Kurtosis'] = k_n
            temp['North_Scintillation_Index'] = si_n

            temp['South_Mean'] = m_s           
            temp['South_STD'] = std_s
            temp['South_Skew'] = s_s
            temp['South_Kurtosis'] = k_s
            temp['South_Scintillation_Index'] = si_s
            
            result[runID] = temp

        return result

    
    
if __name__ == '__main__':
    print('real_accessor_class.py was run as main file.')
    # If needed to recompute and store:
    # summary_stats = mgr.compute_summary_stats()
    # fname = r'summary_stats_dict.pkl'
    # # with open( fname, 'wb' ) as file:
    # #     pickle.dump( summary_stats, file )

    # Otherwise, just load from file:
    # with open(fname, 'rb') as file:
    #     summary_stats = pickle.load(file)   
