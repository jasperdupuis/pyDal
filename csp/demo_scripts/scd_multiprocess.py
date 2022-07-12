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

import matplotlib.pyplot as plt
import numpy as np
    

#my modules in parent directory
sys.path.append('..\..\csp')
import pointwise_scd_function as scd
import data_methods
import parameters as param

hull_sensors = list(param.hull_map_2019.keys()) #2019 DICTIONARY
run_data = dict()

hull_id = hull_sensors[1] # port inboard, should be a good sensor.


df = pd.read_csv(param.df_fname)
#sub select just day 1 results
df_selection = df[df[param.col_burnsi_id].isin(param.run_ids)]

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

#this is for a batch of runs. Missing some post-work.
# for run in param.run_ids:
#     x = run_data[run][hull_id][:param.N_total]
#     samples,timestamps = data_methods.divide_time_series(x,
#                                                          param.overlap_major,
#                                                          param.FS,
#                                                          param.num_seconds_chunk)

#this is for a single run, chosen with the integer argument to run_ids
run = param.run_ids[4]
x = run_data[run][hull_id][:param.N_total]
samples,timestamps = data_methods.divide_time_series(x,
                                                      param.overlap_major,
                                                      param.FS,
                                                      param.num_seconds_chunk)

    
if __name__=='__main__':
    freeze_support()
    samples_iterable = list(samples.T)  #convert 2d array to iterable :
    start = time.time()    
    result = scd.calculate_scd(
        samples_iterable[0],
        param.FS,
        param.bw,
        param.scd_point_freqs,
        param.scd_point_alphas)
    end = time.time()
    print('single unthreaded: ' + str(end - start))
    
    start = time.time()
    nthreads=5
    with Pool(nthreads) as p:
        inputs = samples_iterable
        results = p.map(scd.calculate_scd, inputs)  
        p.close()
        p.join()
    end = time.time()
    print(str(len(samples_iterable)) +' samples with ' \
          + str(nthreads) +' processes: ' + str(end - start))
  
    
    # mess around with results
    res_abs = np.abs(np.array(results))
    plt.figure()
    for sampl in range(res_abs.shape[0]):
        plt.plot(
            param.scd_point_alphas,
            res_abs[sampl,:,10],
            label = param.scd_point_freqs[10]
            )   
    
    
    res_mean = np.mean(res_abs,axis=0)
    plt.figure()
    for index in range(res_mean.shape[1]):
        plt.plot(
            param.scd_point_alphas,
            res_mean[:,index],
            label=param.scd_point_freqs[index])
    plt.legend()
    
    res_mean2 = np.mean(res_mean,axis=1)
    plt.figure()
    plt.plot(param.scd_point_alphas,res_mean2)
