# -*- coding: utf-8 -*-
"""

Messing around with Orca-class datasets and CSP, specifically CMC


"""
DATA = True
COMPUTE = False
PLOT = False

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


hull_sensors = list(param.hull_map_2019.keys())
run_data = dict()
hull_id = hull_sensors[1] # port inboard, should be a good sensor.



if __name__ == '__main__':
    if DATA:
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
            
    if COMPUTE:
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
            t,freqs,alphas,r_cmc,r_spec = cyclic_modulation_coherence.create_sampled_CMC_and_gram(
                                                                     samples,
                                                                     fs = param.FS,
                                                                     M = M,
                                                                     w = w,
                                                                     N_prime = N_prime,
                                                                     L = L,
                                                                     n_alphas = n_alphas,
                                                                     n_freqs = n_freqs)
            
            run_data[run][hull_id + ' Spectrogram frequency (Hz)'] = freqs
            run_data[run][hull_id + ' Cyclic frequency (Hz)'] = alphas
            run_data[run][hull_id + ' Cyclic Modulation Coherence'] = r_cmc
            run_data[run][hull_id + ' Spectrogram'] = r_spec
            
            r_cmc_mean = np.mean(r_cmc,axis=0)
            r_cmc_integrated = np.mean(r_cmc**2,axis=0)
            run_data[run][hull_id + ' Mean Cyclic Modulation Coherence'] = \
                r_cmc_mean 
            run_data[run][hull_id + ' Integrated Cyclic Modulation Coherence'] = \
                np.mean(r_cmc_integrated[:,param.ICMC_MIN:param.ICMC_MAX],axis=1)

    if PLOT:
        data_label = hull_id + ' ' + 'Mean Cyclic Modulation Coherence'
        total_label =  data_label
        x_label = hull_id + ' ' + 'Spectrogram frequency (Hz)'
        y_label = hull_id + ' ' + 'Cyclic frequency (Hz)'
        units = '10log_10(), ref arbitrary'
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
        
        data_label = hull_id + ' ' + 'Integrated Cyclic Modulation Coherence'
        total_label =  data_label
        x_label = hull_id + ' ' + 'Cyclic frequency (Hz)'
        y_label = 'unused 1d'
        units = '10log_10(), ref arbitrary'
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
        

# # ICMC test statistics
# # Tried to implement equation 31 directly but there's an issue
# # with array indexing going out of bounds (still returns values, but mostly =0).
# sigma_alphas = np.zeros_like(alphas)
# for index in range(len(alphas)):
#     alpha = alphas[index]    
#     result = 0
#     for m in range((1-N_prime),N_prime -1):
#         temp = 0
#         for n in range(np.max((0,m)),np.min((N_prime-1,N_prime-1+m))):
#             x1 = np.take(w,n,mode='wrap')
#             x2 = np.take(w,n-m,mode='wrap')
#             temp = temp +  (x1 * x2)
#         temp = np.abs(temp)**2        
#         result = result + ( temp * np.cos(2 * np.pi * alpha * m) )
    
#     sigma_alphas[index] = result 



# #This looks reasonable and comes from analysis in Appendix D of paper.
# w_auto2 = np.abs(signal.correlate(w,w))**2
# wf = np.fft.fft(w_auto2)
# sigma_alphas = wf 



# arg = 1 / ( L * M)
# K = ICMC_MAX-ICMC_MIN # selected freqs
# deg_freedom = 2 * K
# p = 0.05 # antoni's sig level 
# ppf = stats.chi2.ppf(1-p,deg_freedom)
# test_stat = arg * sigma_alphas * ppf / deg_freedom
# wf_freqs = np.fft.fftfreq(len(wf),1/(FS))
# plt.plot(wf_freqs[:20],np.abs(test_stat[:20]))

# # plt.plot(alphas[1:],sigma_alpha[1:])



# plt.plot(wf_freqs[:20],np.abs(test_stat[:20]),label='stat')
# plt.plot(alphas[1:],ICMC[1:],label='ICMC')
# plt.legend()

