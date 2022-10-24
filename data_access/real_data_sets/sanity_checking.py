# -*- coding: utf-8 -*-
"""
Try using my old spectrogram class to see if the issue lies under the 
hood in STFT... thought not though.
(do not have notebook / work stuff to check this rn)
"""


import pandas as pd
import numpy as np
from scipy import interpolate,signal
import matplotlib.pyplot as plt

import hydrophone

#import pydrdc
import sys
sys.path.insert(1, r'C:\Users\Jasper.Dupuis\Desktop\pydrdc')
import signatures


INDEX_FREQ_90000 = 8999

cal_file = r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/TF_DYN_SOUTH_L_40.CSV'
amb_raw_file = r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/ES0451_MOOSE_OTH_DYN/RAW_TIME/AMB_ES0451_DYN_099_005_Shyd_TM.BIN'
amb_range_file = r'C:/Users/Jasper/Desktop/MASC/raw_data/2019-Orca Ranging/Range Data Amalg/ES0451_MOOSE_OTH_DYN/AMB_ES0451_DYN_099_005_Shyd_NB.CSV'

fs = 204800
w = np.hanning(int(fs/10))
overlap = 0

#
#
# Range results
df = pd.read_csv(amb_range_file,skiprows=73)
m_df = df[df.columns[1]]
f_df = df[df.columns[0]]

#
#
# This worked with the range comparison paper (with TL value from range)
range_dictionary = signatures.data.range_info.dynamic_patbay_2019.RANGE_DICTIONARY
hyd = \
    signatures.data.range_hydrophone.Range_Hydrophone_Canada(range_dictionary)
hyd.load_range_specifications(range_dictionary)
uncalibratedDataFloats, labelFinder, message = \
    hyd.load_data_raw_single_hydrophone(amb_raw_file)
uncalibratedDataFloats = uncalibratedDataFloats - np.mean(uncalibratedDataFloats)

spectrogram = signatures.analysis.spectrogram_array.Spectrogram_Array(fs)
spectrogram.set_time_series(uncalibratedDataFloats)
spectrogram.set_window_function(w,
                                'Hanning')
spectrogram.set_overlap_factor(overlap)
spectrogram.create_2d_psd_array()
spectrogram_array = spectrogram.psd_array

#
#
# I would prefer to use this method.
s1 = np.sum(w)
s2 = np.sum(w**2)
f,t,s_z = signal.stft(uncalibratedDataFloats,
                              fs,
                              window = w,
                              nperseg = len(w),
                              noverlap = 0,
                              nfft = None,
                              return_onesided = False)#,
                              # scaling = 'spectrum')
s_z = 2 * (np.abs( s_z )**2 ) / ( fs )  # this should be correct BUT
s_z = s_z * (s1)                     # stft applies 1/s1

# s_z = 2 * (np.abs( s_z )**2) / ( fs )        # STFT already applies (sum(w**2))**0.5

#
#
# old and new methods have different axis - oops. Apply mean along appropriate axis for each.
f_old = spectrogram.psd_freq_basis[1:]
m_old = np.mean(spectrogram.psd_array,axis=0)[1:]
p_old = 10*np.log10(m_old)
f_new = f[1:-1] #need to drop the last element
m_new = np.mean(s_z,axis=1)[1:-1]
p_new = 10*np.log10(m_new)

#
#
# Now need to apply limits up to 90kHz
p_old = p_old[:INDEX_FREQ_90000]
f_old = f_old[:INDEX_FREQ_90000]
p_new = p_new[:INDEX_FREQ_90000]
f_new = f_new[:INDEX_FREQ_90000]    

#
#
# Calibration files from the range
freq_basis_interp = f_new[:INDEX_FREQ_90000] # 0 is out of the interpolation range, also not interested.
cal_s, cal_n = hydrophone.get_and_interpolate_calibrations(freq_basis_interp)



plt.figure()
plt.plot(f_old,p_old+cal_s,label='Old method')
plt.plot(f_new,p_new+cal_s,label = 'New method')
plt.plot(f_df,m_df,label='Range result')
plt.xscale('log')
plt.legend()


d = (p_old+cal_s) - m_df[:INDEX_FREQ_90000]
d = p_old - p_new
np.mean(d)







