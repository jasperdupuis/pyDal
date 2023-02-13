# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 17:12:49 2022

@author: Jasper
"""

from _imports import \
    np,\
    interpolate,\
    os,\
    signatures,\
    signal,\
    h5,\
    pd,\
    RANGE_DICTIONARY

from _directories_and_files import \
    FILE_SOUTH_CAL,\
    FILE_NORTH_CAL,\
    DIR_BINARY_HYDROPHONE, \
    DIR_TRACK_DATA

from _variables import FS_HYD, OOPS_DYN

# the dataframe that holds runIDs and filenames
class Real_Hydrophone():
    
    def __init__(self):
        return
    
    def align_track_and_hyd_data(
        self,
        p_the_run_dictionary,
        labelFinder,
        label_com = 'COM ',
        label_fin = 'FIN ',
        fs_hyd = int(24800),
        t_hyd = 1.5):
        """
        ASSUMPTION: len(hydrophone data time) >= len (gps data time)
        That is, we only need to prune the hydrophone data to match gps data.
        So then in that case there are four cases:
            1) Missing both labels
            2) Missing COM
            3) Missing FIN
            4) Both labels present
        Each case must be treated.
        Further there is the potential that len(hyd_data) / fs > len(gps)/10
        even after truncation!
        In this case find the delta and split it evenly between start and end.    
        
        input dictionary depends on having keys 'North', 'South', 'Time'
        
        labelFinder is the list returned from pydrdc.signature.loadHydData.
        """
        # STEP ONE: Get the labels indices or set them to 0,-1
        try:
            index_com = labelFinder.index(label_com)
        except:
            index_com = 0
        try:
            index_fin = labelFinder.index(label_fin)    
        except:
            index_fin = -1
        # STEP TWO: Apply the label indices to the hydrophone data.
        if index_fin == -1: # Do not want to multiply -1 by fs.
            start = int(index_com * fs_hyd * t_hyd)
            end = int(index_fin)
            p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ start : end ]
            p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ start : end ]
        else: # index IS meaningful, so use it.
            start = int(index_com * fs_hyd * t_hyd)
            end = int(index_fin * fs_hyd * t_hyd)
            p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ start : end ]
            p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ start : end ]
        # STEP THREE: Check if signal lengths are good:
        time_g = p_the_run_dictionary['Time'][-1] - p_the_run_dictionary['Time'][0] # Use this in case samples are missed.
            # Treat time_g for float rounding - only want the first decimal place
        time_g = int(time_g * 10) / 10
        time_h = len(p_the_run_dictionary['North'])/fs_hyd
        if not(time_g == time_h):
            #So, the total hydrophone time is not equal to the total gps time elapsed
            dt = time_h - time_g # +ve ==> hyd time exceeds gps time
            dt = np.round(dt,2)
            trunc_one_ended = int(fs_hyd * dt/2) # amount of data to chop from each end
            p_the_run_dictionary['North'] = p_the_run_dictionary['North'][ trunc_one_ended : -1 * trunc_one_ended ]
            p_the_run_dictionary['South'] = p_the_run_dictionary['South'][ trunc_one_ended : -1 * trunc_one_ended ]
        else:
            # The unlikely case of  gps and hyd times aligning.
            # null operation required
            p_the_run_dictionary['North'] = p_the_run_dictionary['North']        
            p_the_run_dictionary['South'] = p_the_run_dictionary['South']
    
        return p_the_run_dictionary
    
    
    def interpolate_x_y(
        self,
        p_the_run_dictionary):
        # Now, must make sure there is an x,y sample for each time step.
        # Note ther eare missing time steps but we know they occured, so 
        # interpolate away!
        # 2x 1d interpolations for each of x, y
        x_function = interpolate.interp1d(
            p_the_run_dictionary['Time'], # x
            p_the_run_dictionary['X'])    # f(x)
        y_function = interpolate.interp1d(
            p_the_run_dictionary['Time'], # x
            p_the_run_dictionary['Y'])    # f(x)
        t = np.arange(
            p_the_run_dictionary['Time'][0], #start
            p_the_run_dictionary['Time'][-1], #stop
            1/FS_GPS)                        #step
        p_the_run_dictionary['X'] = x_function(t)
        p_the_run_dictionary['Y'] = y_function(t)
        p_the_run_dictionary['Time'] = t
        return p_the_run_dictionary
    
    
    def get_and_interpolate_calibrations(
            self,
            p_target_f_basis = np.arange(10,9e4,10),
            p_hyd_s = FILE_SOUTH_CAL,
            p_hyd_n = FILE_NORTH_CAL,
            p_target_bw = 10, # Hz
            p_df_nb = 2/3,
            # p_fname_n = r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/TF_DYN_NORTH_L_40.CSV',
            # p_fname_s = r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/TF_DYN_SOUTH_L_40.CSV',
            p_range_dictionary = RANGE_DICTIONARY
            ):
        
        # interpolate range calibration file
        # The assumption is that variation is pretty slow in bandwidths of itnerest
        # That is below say 15Hz. Quick plot shows this is true.
    
        df_s = pd.read_csv(FILE_SOUTH_CAL,
                           skiprows=p_range_dictionary['AMB CAL South Spectral file lines to skip'],
                           encoding = "ISO-8859-1")
        df_n = pd.read_csv(FILE_NORTH_CAL,
                           skiprows=p_range_dictionary['AMB CAL North Spectral file lines to skip'],
                           encoding = "ISO-8859-1")
        freqs = df_s[df_s.columns[0]].values
        len_conv = int(p_target_bw / p_df_nb)
        s = df_s[df_s.columns[1]].values # Should be AMPL (which is really dB)
        n = df_n[df_n.columns[1]].values # SHould be AMPL (which is really dB)
        # Valid provides results only where signals totally overlap
        sc = np.convolve( s, np.ones(len_conv)/len_conv, mode='valid')
        nc = np.convolve( n, np.ones(len_conv)/len_conv, mode='valid')
        # convoluton chops a bit; append values at the end where change is not interesting.
        delta = np.abs( len( sc ) - len( s ) ) # number of missing samples to add; always -ve so take abs
        last = sc[ -1 ] * np.ones(delta)
        sc = np.append(sc,last)
        nc = np.append(nc,last)
        sfx = interpolate.interp1d( freqs, sc ) #lazy way to do it.
        nfx = interpolate.interp1d( freqs, nc )
        ncal = nfx(p_target_f_basis) # these are the results
        scal = sfx(p_target_f_basis) # these are the results
        
        return scal,ncal
    
    
    def generate_files_from_runID_list(
        self,
        p_list_run_IDs,
        p_df,  
        p_fs_hyd = FS_HYD,
        p_window = np.hanning(204800),
        p_overlap_n = 0,
        p_rel_dir_name = '',
        p_range_dictionary = RANGE_DICTIONARY,
        p_trial_search = 'DRJ',
        p_binary_dir = DIR_BINARY_HYDROPHONE,
        p_track_dir = DIR_TRACK_DATA,
        mistakes = OOPS_DYN):
        """
        Can be made to work with any SRJ/DRJ/DRF/SRF etc run ID substring, 
        uses contain so needn't necessarily be the front.
        2019 and 2022 have different dir structures so must be provided.
        
        Stores results as linear arrays!
        """
        for runID in p_list_run_IDs:
            if not (p_trial_search == runID[:3]): 
                continue #only want provided data initiator (SRJ, DRJ, AMJ, etc).
            if runID in mistakes: 
                continue #I  made some mistakes... must reload these trk files properly later
            fname_hdf5 = p_rel_dir_name + r'\\'+ runID + r'_data_timeseries.hdf5'           
            if os.path.exists(fname_hdf5): 
                continue # If the hdf5 file already exists, no need to do it.
            temp = dict()
            row = p_df[ p_df ['Run ID'] == runID ]
            
            fname = p_binary_dir + row['South hydrophone raw'].values[0]
            hyd = \
                signatures.data.range_hydrophone.Range_Hydrophone_Canada(p_range_dictionary)
            hyd.load_range_specifications(p_range_dictionary)
            uncalibratedDataFloats_south, labelFinder, message = hyd.load_data_raw_single_hydrophone(fname)
            temp['South'] = uncalibratedDataFloats_south
            
            fname = p_binary_dir + row['North hydrophone raw'].values[0]
            hyd = \
                signatures.data.range_hydrophone.Range_Hydrophone_Canada(p_range_dictionary)
            hyd.load_range_specifications(p_range_dictionary)
            uncalibratedDataFloats_north, labelFinder, message = hyd.load_data_raw_single_hydrophone(fname)
            temp['North'] = uncalibratedDataFloats_north
            
            if runID[:2] == 'DR': # track only matters for dynamic
                fname = p_track_dir + row['Tracking file'].values[0]
                track = signatures.data.range_track.Range_Track()
                track.load_process_specifications(p_range_dictionary)
                track.load_data_track(fname)
                start_s_since_midnight, total_s = \
                    track.trim_track_data(r = RANGE_DICTIONARY['Track Length (m)'] / 2,
                        prop_x_string = RANGE_DICTIONARY['Propeller X string'],
                        prop_y_string = RANGE_DICTIONARY['Propeller Y string'],
                        CPA_X = RANGE_DICTIONARY['CPA X (m)'],
                        CPA_Y = RANGE_DICTIONARY['CPA Y (m)'])
                df_temp = track.data_track_df_trimmed
                    
                temp['X'] = df_temp[ RANGE_DICTIONARY['Propeller X string'] ].values
                temp['Y'] = df_temp[ RANGE_DICTIONARY['Propeller Y string'] ].values
                temp['Time'] = df_temp[ RANGE_DICTIONARY['Time string'] ].values
            
                temp = self.align_track_and_hyd_data(temp, labelFinder) # do some truncation
                temp = self.interpolate_x_y(temp) # make sure the entire time base is represented
                
            s1 = np.sum(p_window)
            s2 = np.sum(p_window**2) # completeness - not used by me. STFT applies it.
            #Now the 'grams
            f,s_t,s_z = signal.stft(temp['South'],
                                  p_fs_hyd,
                                  window = p_window,
                                  nperseg = len(p_window),
                                  noverlap = p_overlap_n,
                                  nfft = None,
                                  return_onesided = True)
            f,n_t,n_z = signal.stft(temp['North'],
                                  p_fs_hyd,
                                  window = p_window,
                                  nperseg = len(p_window),
                                  noverlap = p_overlap_n,
                                  nfft = None,
                                  return_onesided = True)
            s_z = 2 * (np.abs( s_z )**2) / ( s2)        # PSD
            s_z = s_z * s1                              # stft applies 1/s1, reverse this
            n_z = 2 * (np.abs( n_z )**2) / ( s2)        # PSD
            n_z = n_z * (s1)                            # stft applies 1/s1, reverse this
            temp['South_Spectrogram'] = s_z
            temp['North_Spectrogram'] = n_z
            temp['South_Spectrogram_Time'] = s_t
            temp['North_Spectrogram_Time'] = n_t
            temp['Frequency'] = f
            
            try:
                os.remove(fname_hdf5)
            except:
                print(runID + ' hdf5 file did not exist before generation')
                
            with h5.File(fname_hdf5, 'w') as file:
                for data_type,data in temp.items():
                    # note that not all variable types are supported but string and int are
                    file[data_type] = data


if __name__ == "__main__":    
    import pandas as pd
    import numpy as np
    FS_GPS = 10
    FS_HYD = 204800
    trial_runs_file = 'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'
    trial_binary_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'
    trial_track_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\TRACKING\\'
    
    local_df = pd.read_csv(trial_runs_file)
    list_run_IDs = local_df[ local_df.columns[1] ].values
    list_runs = list_run_IDs # this might not be a good idea...
    #
    # The below processes the dynamic data from 2019 trial. (DRJ)
    #
    rel_dir_name = 'hdf5_timeseries_bw_01_overlap_90'

    overlap = 0.9
    overlap_n = FS_HYD * overlap
    window = np.hanning(FS_HYD)
    df = local_df
    fs_hyd = FS_HYD
    fs_gps = FS_GPS
    range_dict = RANGE_DICTIONARY
    trial_search = 'DRJ'
    
    hydro = Real_Hydrophone()
    hydro.generate_files_from_runID_list(list_runs,df,fs_hyd,window,overlap_n,rel_dir_name,range_dict,trial_search)
