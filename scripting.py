# -*- coding: utf-8 -*-
"""

a file for the Buckleys of code:
    It's ugly, but it's works
    
"""


import _variables


the_location= Location(variables.LOCATION)

list_files = os.listdir(variables.DIR_SPECTROGRAM)
list_runs = [x.split('_')[0] for x in list_files]
list_runs = [ x for x in list_runs if not('summary' in x)]
mgr = mgr_real(list_runs,variables.DIR_SPECTROGRAM)    


summary_stats = mgr.load_and_set_pickle_summary_stats(
    variables.DIR_SPECTROGRAM + variables.SUMMARY_FNAME)

list_ambients = [ x for x in list_runs if 'AM' in x ]

runID = list_ambients[1]
hyd = 'North'

key_kurtosis = hyd + '_Kurtosis'
key_scint = hyd + '_Scintillation_Index'
key_skew = hyd + '_Skew'
key_std = hyd + '_STD'

freq    = summary_stats[runID]['Frequency']
index_1khz = np.max(np.where(freq<1000))
kurt    = summary_stats[runID][key_kurtosis]
scint   = summary_stats[runID][key_scint]
skew    = summary_stats[runID][key_skew]
std     = summary_stats[runID][key_std]

plt.figure()
for runID in list_runs:
    if runID in list_ambients: continue
    if 'DRJ3' not in runID: continue
    freq    = summary_stats[runID]['Frequency']
    index_1khz = np.max(np.where(freq<1000))
    kurt    = summary_stats[runID][key_kurtosis]
    scint   = summary_stats[runID][key_scint]
    skew    = summary_stats[runID][key_skew]
    std     = summary_stats[runID][key_std]

    plt.plot(freq[:index_1khz],kurt[:index_1khz],label=runID);plt.xscale('log')
plt.legend()



