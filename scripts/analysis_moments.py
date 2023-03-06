# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:37:10 2023

@author: Jasper
"""

from _imports import \
    Moment_Analyzer,\
    np,\
    plt

analyzer = Moment_Analyzer()
analyzer.load_summary_stats()

stats = analyzer.mgr.summary_stats
keys_all = stats.keys() 
keys_amb = [x for x in keys_all if 'AMJ' in x]
key_any = keys_amb[1]
#find the number of runs in the ambient keys
n = 0
for key in keys_amb:
    n = n + 1

# make a results dictionary that makes an array for each key type
result = dict()
for k,v in stats[key_any].items():
    if 'Freq' in k: continue
    result[k] = np.zeros([len(v),n])

#now insert the values in to array
index = 0
for k,v in stats.items(): # for each run in the summary stats dictionary
    if not(k in keys_amb): continue #only want ambients
    for kk,vv in stats[k].items(): # each summary stat type in current run
        if 'Freq' in kk: continue
        result[kk][:,index] = vv
    index = index + 1
                  
freq = stats[key]['Frequency']
for k,v in result.items():
    if 'freq' in k: continue
    mean = np.mean(v,axis = 1)
    std = np.std(v,axis=1)
    plt.figure()
    plt.plot(freq,std);plt.xscale('log')
    plt.title(k)


