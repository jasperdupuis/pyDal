# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:02:21 2022

@author: Jasper
"""


from _imports import h5, np, plt, stats, pickle, pd
from _imports import  Real_Amb, Real_Hyd, os, Location

from _variables import GOOD_AMB, LOCATION

import _variables

import _directories_and_files

TYPE = 'linear'

class Real_Data_Manager():
    """
    This class should be able to simplify life, e.g.:
        process raw data to hdf5 formats - CHECK
        get hydrophone time series - CHECK
        get hydrophone ambients - CHECK
        get hydrophone spectrograms - CHECK
        retreieve spectrogram data for a target frequency - CHECK 
        retrieve all ambient data for a target frequency - CHECK
        facilitate basic regression analysis
        facilitate future translation of this data in to ML-ready data.
    """
    
    def __init__(self,
                 p_list_runs,
                 p_data_dir):
        
        self.ambient_obj = Real_Amb()
        self.hydro_obj = Real_Hyd()
        
        self.list_runs = p_list_runs
        self.data_dir = p_data_dir
        
        #Hard coded variables below, probably dont need to touch them.
        
        # These are for  1.0s windows
        # self.INDEX_FREQ_LOW = 3
        # self.INDEX_FREQ_HIGH = 89999 #90k cutoff
        
        # These are for 0.1 s windows
        self.INDEX_FREQ_LOW = 1
        self.INDEX_FREQ_HIGH = 8999 #90k cutoff]
            
        # self.FS_HYD = 204800
        # self.T_HYD = 1.5 #window length in seconds
        # self.FS_GPS = 10
        # self.LABEL_COM = 'COM '
        # self.LABEL_FIN = 'FIN '
        
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
        freq = self.ambient_obj.get_freq_basis(
            self.list_runs[0],
            p_index_low  =  self.INDEX_FREQ_LOW,
            p_index_high = self.INDEX_FREQ_HIGH,
            p_data_dir = self.data_dir)
        
        self.freq_basis_trimmed = freq
    

    def load_calibrations(self):
            s,n = self.hydro_obj.get_and_interpolate_calibrations(self.freq_basis_trimmed)
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
               result['North_Spectrogram']      = file['North_Spectrogram'][:]
               result['North_Spectrogram_Time'] = file['North_Spectrogram_Time'][:]
               result['South_Spectrogram']      = file['South_Spectrogram'][:]
               result['South_Spectrogram_Time'] = file['South_Spectrogram_Time'][:]
               result['Frequency']              = file['Frequency'][:]
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
        FOR NOW, set cal_s and cal_n to be zero here to easily compare
        """
        amb_s,amb_n,runs_used,freqs_ret = \
            self.ambient_obj.return_ambient_results( 
                                            GOOD_AMB, #Not all ambient runs are good. # NOT HANDLED WELL RN
                                            np.zeros_like(self.cal_n),
                                            np.zeros_like(self.cal_s),
                                            self.INDEX_FREQ_LOW,
                                            self.INDEX_FREQ_HIGH,
                                            self.data_dir)
        self.amb_s = amb_s
        self.amb_n = amb_n
        self.amb_runs = runs_used


    def plot_all_ambients(self,p_f_high=1000):
        index_high = np.where(self.freq_basis_trimmed > p_f_high)[0][0]
        fig_n,ax_n = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        for r,a in zip(self.amb_runs,self.amb_n):
            ax_n.plot(
                self.freq_basis_trimmed[:index_high],
                a[:index_high],
                label = r)
        ax_n.set_xscale('log')
        fig_n.supylabel('dB ref 1 V^2 / Hz')
        fig_n.supxlabel('Frequency (Hz)')
        fig_n.suptitle('North Hydrophone Ambients')
        fig_n.legend()

        fig_s,ax_s = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        for r,a in zip(self.amb_runs,self.amb_s):
            ax_s.plot(
                self.freq_basis_trimmed[:index_high],
                a[:index_high],
                label = r)
        ax_s.set_xscale('log')
        fig_s.supylabel('dB ref 1 V^2 / Hz')
        fig_s.supxlabel('Frequency (Hz)')
        fig_s.suptitle('South Hydrophone Ambients')
        fig_s.legend()

        return fig_n, ax_n, fig_s, ax_s
                

    def extract_target_frequency(self,
                                 p_run_data_dict,
                                 p_target_index):
        samp_n = p_run_data_dict['North_Spectrogram'][p_target_index,:]
        samp_s = p_run_data_dict['South_Spectrogram'][p_target_index,:]
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
                                       p_runs,
                                       p_type, #Ambient AM or dynamic DR
                                       p_target_freq,
                                       p_day = 'DRJ3',
                                       p_speed = '07',
                                       p_hyd = 'NORTH',
                                       p_decibel_bool = True,
                                       p_ambients_bool =True):
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
        for runID in p_runs:
            if (p_day not in runID): continue # this run is not the right day
            if (p_speed not in runID): continue # this run is not the right speed
            if 'frequency' in runID: continue # this is not a run.
            if 'summary' in runID: continue # this is not a run
            
            runData = self.load_target_spectrogram_data(runID,  # Returns linear values
                                                       self.data_dir)
            samp_n,t_n,samp_s,t_s = self.extract_target_frequency(runData,
                                                                 target_f_index)
            if p_decibel_bool:
                samp_n = 10*np.log10(samp_n)
                samp_s = 10*np.log10(samp_s)
                
            if p_type == 'DR' :
                #Ambient wont have this data, treat in next if
                x = runData['X'][:]
                y = runData['Y'][:]
                r = np.sqrt(x*x + y*y) 
                index_cpa = np.where(r==np.min(r))[0][0]
                
                if p_hyd == 'NORTH':
                    ax = self.scatter_time_series(t_n, samp_n, ax, label=runID)
                    t_n = t_n-np.min(t_n)
                    plt.axvline( t_n [ index_cpa ] )
            
                if p_hyd == 'SOUTH':
                    ax = self.scatter_time_series(t_s, samp_s, ax, label=runID)
                    t_s = t_s-np.min(t_s)
                    plt.axvline( t_s [ index_cpa ] )
                    
            if p_type =='AM' :
                if p_hyd == 'NORTH':
                    ax = self.scatter_time_series(t_n, samp_n, ax, label=runID)
                    t_n = t_n-np.min(t_n)
                    
                if p_hyd == 'SOUTH':
                    ax = self.scatter_time_series(t_s, samp_s, ax, label=runID)
                    t_s = t_s-np.min(t_s)
                    
        if p_ambients_bool:
            ax = self.plot_ambient_level_single_f(ax)
        plt.title(str(target_freq) + ' Hz with ambient received levels as horizontal lines \n 1 Hz BW, db ref V^2')
        plt.legend()
        return fig,ax


    def compute_summary_stats(self,p_type=TYPE):
        """
        Compute mean, std, SI, S, K for each frequency bin and run.
        
        pass p_type = 'decibel' for 10log10(value)
        """
        result = dict()
        for runID in self.list_runs:
            if 'summary' in runID: continue # not valid.
            temp = dict()
            if 'frequency' in runID: continue # don't want this run.
            runData = self.load_target_spectrogram_data(runID,  # Returns linear values
                                                       self.data_dir)
            spec_n = \
                runData['South_Spectrogram'][self.INDEX_FREQ_LOW : self.INDEX_FREQ_HIGH , : ]
            spec_s = \
                runData['North_Spectrogram'][self.INDEX_FREQ_LOW : self.INDEX_FREQ_HIGH , : ]
            
            if p_type == 'decibel':
                spec_n = 10*np.log10(spec_n)
                spec_s = 10*np.log10(spec_s)
                
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
            
            temp['Frequency'] = \
                runData['Frequency'][self.INDEX_FREQ_LOW : self.INDEX_FREQ_HIGH ]
            
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
    
    
    def compute_and_pickle_summary_stats(
            self,
            fname,
            p_type = 'linear'):
        dict_summary_stats = self.compute_summary_stats(p_type)
        with open(fname, 'wb') as f:
            pickle.dump(dict_summary_stats, f)
         
        
    def load_and_set_pickle_summary_stats(self,fname):
        with open(fname, 'rb') as f:
            summary_stats = pickle.load(f)
            self.summary_stats = summary_stats
            return summary_stats
        
        
    def plot_skew_kurtosis_SI_for_run(self,p_runID,p_side = 'North'):
        """
        Must have loaded the summary_stats from
        load_and_set_pickle_summary_stats
        # Usage:
        fig = mgr.plot_skew_kurtosis_SI_for_run(TEST_RUN)
        plt.show()

        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(12,7))
        fig.suptitle(p_runID + ': Skew, Kurtosis, and Scintillation Index')
        r = self.summary_stats[p_runID]
        
        ax1.plot(self.freq_basis_trimmed,r[p_side + '_Skew']);
        ax1.set_xscale('log')
        ax1.title.set_text('Skew')
        
        ax2.plot(self.freq_basis_trimmed,r[p_side + '_Kurtosis']);
        ax2.set_xscale('log')
        ax2.title.set_text('Kurtosis')       
        
        ax3.plot(self.freq_basis_trimmed,r[p_side + '_Scintillation_Index']);
        ax3.set_xscale('log')
        ax3.title.set_text('Scintillation Index')
        
        return fig

    def get_config_dictionary(self,p_fname):
        with open(p_fname, 'rb') as f:
            config_run_dictionary = pickle.load(f)
            return config_run_dictionary
        
    def set_rpm_table(self,p_fname):
        df= pd.read_csv(p_fname,sep = ' ')    
        hzs = df[df.columns[2]].values / 60
        df['Corrected RPM as Hz'] = hzs
        self.df_speed_rpm = df
        
    def return_nominal_rpm_as_hz(self,p_speed):
        hz = \
            self.df_speed_rpm [self.df_speed_rpm ['Speed(kn)']==3][self.df_speed_rpm.columns[3]].values[0]
        return hz

    
    @staticmethod
    def get_manager_and_runs(
            p_dir_spectrograms = _directories_and_files.DIR_SPECTROGRAM,
            p_fname_summary = _directories_and_files.SUMMARY_FNAME,
            p_fname_trial_map = _directories_and_files.FNAME_CONFIG_TRIAL_MAP):
    
        list_files = os.listdir(p_dir_spectrograms)
        list_runs = [x.split('_')[0] for x in list_files]
        mgr = Real_Data_Manager(list_runs,p_dir_spectrograms)    
    
        summary_stats = \
            mgr.load_and_set_pickle_summary_stats(
            p_dir_spectrograms + p_fname_summary )
    
        config_dictionary = \
            mgr.get_config_dictionary(
                p_fname_trial_map )
            
        #Selects based on run ID criteria
        runs = mgr.get_run_selection(
                                    p_type = _variables.TYPE,
                                    p_mth = _variables.MTH,
                                    p_machine = _variables.STATE,
                                    p_speed = _variables.SPEED,
                                    p_head = _variables.HEADING)
            
        return mgr, runs, config_dictionary



if __name__ == '__main__':
    print('real_accessor_class.py was run as main file.')
    # # If needed to recompute and store:
    # import sys,pickle
    # sys.path.append('C:\\Users\\Jasper\\Documents\\Repo\\pyDal\\pyDal')

    # from _imports import Location, os
    # from _directories_and_files import DIR_SPECTROGRAM
    # from _variables import LOCATION    

    # the_location = Location(LOCATION)
    # list_files = os.listdir(DIR_SPECTROGRAM)
    # list_runs = [x.split('_')[0] for x in list_files]

    # mgr = Real_Data_Manager(list_runs, DIR_SPECTROGRAM)
    # summary_stats = mgr.compute_summary_stats()
    # fname = DIR_SPECTROGRAM + r'summary_stats_dict_decibel.pkl'
    # with open( fname, 'wb' ) as file:
    #     pickle.dump( summary_stats, file )

    # # # Otherwise, just load from file:
    # with open(fname, 'rb') as file:
    #     summary_stats = pickle.load(file)   
