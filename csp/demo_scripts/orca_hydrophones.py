# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:35:23 2022

@author: Jasper
"""

#core modules
import sys
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import nptdms
import pandas as pd
import matplotlib.pyplot as plt

#my modules in parent directory
sys.path.append('..\..\csp')
import cyclic_modulation_coherence
import data_methods
import util_plotting                
import parameters as param # FS=f_s, chunk sizes, overlaps, etc live here.

#import pydrdc for range binary.
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures


RANGE_DICTIONARY = signatures.data.range_info.dynamic_patbay_2019.RANGE_DICTIONARY


raw_label = 'Uncalibrated voltage'
label_label = 'Chunk labels'
message_label = 'Chunk messages'

#no sensor differentiation here, a hydro is a hydro, nominally.

df = pd.read_csv(param.df_fname)
#sub select just day 1 results
df_selection = df[df[param.col_burnsi_id].str.contains('DRJ1')]

run_data = dict()
for index,row in df_selection.iterrows():
    print (row['Run ID'])
    if not (row['Run ID'] in param.run_ids):  #skip this identifier
        continue
    full_path = \
        param.hydro_dir + row[param.col_hydro_file]
    hydro = \
        signatures.data.range_hydrophone.Range_Hydrophone_Canada(
            param.FS,
            1/param.del_f)
    hydro.load_range_specifications(RANGE_DICTIONARY)
    data_uncal, labels, messages = hydro.load_data_raw_single_hydrophone(full_path)
    hydro_data = { raw_label : data_uncal,
                  label_label : labels,
                  message_label : messages}
    run_data[row[param.col_burnsi_id]] = hydro_data
    

for run in param.run_ids:
    midpoint = len(run_data[run][raw_label])//2
    start = midpoint - param.N_total//2
    end = midpoint + param.N_total//2
    x = run_data[run][raw_label][start:end]
    temp_dict = dict() #largest effect on del_alpha and max alpha

    N_prime,L,M,del_alpha_achieved,n_freqs,n_alphas,w = \
        cyclic_modulation_coherence.calculate_parameters(
            N = param.N_chunk,
            fs = param.FS,
            del_f = param.del_f,
            overlap_minor = param.overlap_minor)

    samples,timestamps = \
        data_methods.divide_time_series(
            x,
            param.overlap_major,
            param.FS,
            param.num_seconds_chunk)
    t,freqs,alphas,r_cmc,r_spec = \
        cyclic_modulation_coherence.create_sampled_CMC_and_gram(
            samples,
            fs = param.FS,
            M = M,
            w = w,
            N_prime = N_prime,
            L = L,
            n_alphas = n_alphas,
            n_freqs = n_freqs)

    run_data[run]['Spectrogram frequency (Hz)'] = freqs
    run_data[run]['Cyclic frequency (Hz)'] = alphas
    run_data[run]['Cyclic Modulation Coherence'] = r_cmc
    run_data[run]['Spectrogram'] = r_spec

    r_cmc_mean = np.mean(r_cmc,axis=0)
    r_cmc_integrated = np.mean(r_cmc**2,axis=0)
    run_data[run]['Mean Cyclic Modulation Coherence'] = \
        r_cmc_mean 
    run_data[run]['Integrated Cyclic Modulation Coherence'] = \
        np.mean(r_cmc_integrated[:,param.ICMC_MIN:param.ICMC_MAX],axis=1)


data_label = 'Mean Cyclic Modulation Coherence'
total_label =  data_label
x_label = 'Spectrogram frequency (Hz)'
y_label = 'Cyclic frequency (Hz)'
units = 'arbitrary'
shape = (2,4)
subhead = ''
fig = util_plotting.plot_multi_data(
        run_data,
        param.run_ids, 
        p_data_ref = data_label, 
        p_x_ref = x_label,
        p_y_ref = y_label,
        p_shape = shape,
        p_subheading = subhead, 
        p_units = 'arbitrary', 
        p_xlims = [0,0], 
        p_ylims = [0,0], 
        p_linear=True
        )


data_label = 'Integrated Cyclic Modulation Coherence'
total_label =  data_label
x_label = 'Cyclic frequency (Hz)'
y_label = 'unused 1d'
units = 'arbitrary'
shape = (2,4)
subhead = ''
fig = util_plotting.plot_multi_data(
        run_data,
        param.run_ids, 
        p_data_ref = data_label, 
        p_x_ref = x_label,
        p_y_ref = y_label,
        p_shape = shape,
        p_subheading = subhead, 
        p_units = 'arbitrary', 
        p_xlims = [0,0], 
        p_ylims = [0,0], 
        p_linear=True
        )

