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

#import pydrdc for range binary.
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures


FS = 204800
RANGE_DICTIONARY = signatures.data.range_info.dynamic_patbay_2019.RANGE_DICTIONARY


num_seconds_chunk = 5
num_seconds_total = 15
N_chunk = FS * num_seconds_chunk
N_total = FS * num_seconds_total #largest effect on del_alpha and max alpha
overlap_major = 0 # Overlap of chunks to make a single gram
overlap_minor = 0.75 # Overlap of windows used in making a single gram, often denoted L in integer form
del_f = 25 # a major determinant in how much cyclic resolution you get. Higher delf ==> more alpha.

ICMC_MIN = 41 #index
ICMC_MAX = 410 #index

col_hydro_file = 'South hydrophone raw'
col_burnsi_id = 'Run ID'

hydro_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'
df_fname = r'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'

raw_label = 'Uncalibrated voltage'
label_label = 'Chunk labels'
message_label = 'Chunk messages'

#no sensor differentiation here, a hydro is a hydro, nominally.
run_ids = ['DRJ1PB03AX00EB',
'DRJ1PB05AX00EB',
'DRJ1PB07AX00WB',
'DRJ1PB09AX00WB',
'DRJ1PB11AX00WB',
'DRJ1PB13AX00WB',
'DRJ1PB15AX00WB',
'DRJ1PB17AX00WB']


df = pd.read_csv(df_fname)
#sub select just day 1 results
df_selection = df[df[col_burnsi_id].str.contains('DRJ1')]

run_data = dict()
for index,row in df_selection.iterrows():
    print (row['Run ID'])
    if not (row['Run ID'] in run_ids):  #skip this identifier
        continue
    full_path = hydro_dir + row[col_hydro_file]
    hydro = signatures.data.range_hydrophone.Range_Hydrophone_Canada(FS,1/del_f)
    hydro.load_range_specifications(RANGE_DICTIONARY)
    data_uncal, labels, messages = hydro.load_data_raw_single_hydrophone(full_path)
    hydro_data = { raw_label : data_uncal,
                 label_label : labels,
                 message_label : messages}
    run_data[row[col_burnsi_id]] = hydro_data
    

for run in run_ids:
    midpoint = len(run_data[run][raw_label])//2
    start = midpoint - N_total//2
    end = midpoint + N_total//2
    x = run_data[run][raw_label][start:end]
    temp_dict = dict() #largest effect on del_alpha and max alpha

    N_prime,L,M,del_alpha_achieved,n_freqs,n_alphas,w = \
        cyclic_modulation_coherence.calculate_parameters(
            N = N_chunk,
            fs = FS,
            del_f = del_f,
            overlap_minor = overlap_minor)

    samples,timestamps = data_methods.divide_time_series(x,overlap_major,FS,num_seconds_chunk)
    t,freqs,alphas,r_cmc,r_spec = cyclic_modulation_coherence.create_sampled_CMC_and_gram(
                                                            samples,
                                                             fs = FS,
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



### PLOT CMC
nrow = 2
ncol = 4
selectors = []
for row in range(nrow):
    for col in range(ncol):
       selectors.append((row,col)) 
f,ax_arr = plt.subplots(nrow,ncol)
index = 0
data_label = 'Cyclic Modulation Coherence'
total_label =  data_label
x_label = 'Spectrogram frequency (Hz)'
y_label = 'Cyclic frequency (Hz)'
units = '10log_10(), ref arbitrary'
for index in range(len(run_ids)):
    run = run_ids[index]
    label = run
    _,im = util_plotting.plot_and_return_2D_axis(
        ax_arr[selectors[index]],
        np.mean(run_data[run][data_label],axis=0), #3 is CMC, 4 is PSD
        run_data[run][x_label],
        run_data[run][y_label],
        p_x_min = 0,
        p_x_max = run_data[run][x_label][-1],
        p_y_min = 0,
        p_y_max = run_data[run][y_label].shape[0],
        p_label = label,
        p_linear=False)
    index+= 1
# f.supylabel('Spectrogram time (s)')     #FUTURE RELEASE OF MATPLOTLIB
# f.supxlabel('Spectrogram frequency (Hz)') # FUTURE RELEASE OF MATPLOTLIB
f.text(0.5, 0.04, x_label, ha='center')
f.text(0.04, 0.5, y_label, va='center', rotation='vertical')
cbar = f.colorbar(im, ax=ax_arr.ravel().tolist())
cbar.ax.set_ylabel(units)
f.suptitle(total_label + '\n' + 'July 2019 Orca-class {'+col_hydro_file+'}')
# f.show()


### PLOT ICMC
f,ax_arr = plt.subplots(nrow,ncol)
index = 0
data_label = 'Cyclic Modulation Coherence'
total_label = 'Integrated Cyclic Modulation Coherence'
x_label = 'Cyclic frequency (Hz)'
y_label = 'ICMC magnitude'
units = '10log_10(), ref arbitrary'
for index in range(len(run_ids)):
    run = run_ids[index]
    label = run
    ICMC = np.mean(run_data[run][data_label]**2,axis=0) ##square, then average along chunk
    ICMC = np.mean(ICMC[:,ICMC_MIN:ICMC_MAX],axis=1)/(ICMC_MAX-ICMC_MIN) #average along freq
    _,im = util_plotting.plot_and_return_1D_axis(
        ax_arr[selectors[index]],
        ICMC  ,  
        run_data[run][x_label],
        p_x_min = 0,
        p_x_max = run_data[run][x_label][-1],
        p_label = label,
        p_linear=True)
    
    index+= 1
# f.supylabel('Spectrogram time (s)')     #FUTURE RELEASE OF MATPLOTLIB
# f.supxlabel('Spectrogram frequency (Hz)') # FUTURE RELEASE OF MATPLOTLIB
f.text(0.5, 0.04, x_label, ha='center')
f.text(0.04, 0.5, y_label, va='center', rotation='vertical')
f.suptitle(total_label + '\n' + 'July 2019 Orca-class {'+col_hydro_file +'}')
# f.show()
