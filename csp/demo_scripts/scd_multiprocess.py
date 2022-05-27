# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:39:35 2022

@author: Jasper
"""

#core modules
from multiprocessing import Pool,freeze_support
import sys
import time
import nptdms
import pandas as pd

#my modules in parent directory
sys.path.append('..\..\csp')
import pointwise_scd_function
import data_methods
import parameters as param

tdms_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\AllTDMS\\'
df_fname = r'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'
hull_sensors = list(param.hull_map_2019.keys()) #2019 DICTIONARY
run_data = dict()

hull_id = hull_sensors[1] # port inboard, should be a good sensor.


df = pd.read_csv(df_fname)
#sub select just day 1 results
df_selection = df[df[param.col_burnsi_id].str.contains('DRJ1')]

for index,row in df_selection.iterrows():
    hull_data = dict()
    td = nptdms.TdmsFile.read(param.tdms_dir + row[param.col_tdms_file])
    group = td.groups()[0]
    channels = group.channels()
    for c in channels:
        if len(c.properties) < 5 : #Channel unused on trial
            continue
        if c.properties['ID'] in hull_sensors:
            hull_data[c.properties['ID']] = c.data
    run_data[row[param.col_burnsi_id]] = hull_data
    td.close()

for run in param.run_ids:
    x = run_data[run][hull_id][:param.N_total]
    temp_dict = dict() #largest effect on del_alpha and max alpha

    N_prime,L,M,del_alpha_achieved,n_freqs,n_alphas,w = \
        cyclic_modulation_coherence.calculate_parameters(
            N = param.N_chunk,
            fs = param.FS,
            del_f = param.del_f,
            overlap_minor = param.overlap_minor)

    samples,timestamps = data_methods.divide_time_series(x,
                                                         param.overlap_major,
                                                         param.FS,
                                                         param.num_seconds_chunk)
    
if __name__=='__main__':
    freeze_support()
    #convert samples back to iterable:
    samples_iterable = list(samples.T)
    start = time.time()    
    result = calculate_scd(samples[0],t,FS,bw,freqs,alphas)
    end = time.time()
    print('single unthreaded: ' + str(end - start))
    
    start = time.time()
    with Pool(10) as p:
        inputs = [x1,x2,x3]
        results = p.map(calculate_scd, samples)    
    end = time.time()
    print('three multithreaded: ' + str(end - start))
  

