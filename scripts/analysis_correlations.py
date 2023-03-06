import _variables

import _directories_and_files

from _imports import \
    Real_Hyd,\
    Real_Amb,\
    Real_Data_Manager,\
    Bathymetry_CHS_2,\
    Location,\
    np,\
    stats,\
    plt
    
def run_correlations():
    """
    Compute correlation and p-value for said correlation for
    each frequency spectrum line in 
    specs v time
    specs v 20logR
    specs v specs (n v s)
    
    all in linear domain except v 20 logR as that doesn't otherwise make sense.
    """
    
    mgr, list_runs, config_dictionary = \
        Real_Data_Manager.get_manager_and_runs() #has default args defined.
    
    results_dict = dict()
    for run in list_runs:
        temp = dict()  # each run will have results put here
        spec_dict_result = \
            mgr.load_target_spectrogram_data(
                run, 
                _directories_and_files.DIR_SPECTROGRAM)
    
        f = spec_dict_result['Frequency']
        r = spec_dict_result['R']   # distance from CPA
        r = np.sqrt (r**2 + 100**2) # distance from hydrophone
      
        # time, a proxy for x-y.
        corr_n = np.zeros_like(f)
        p_n = np.zeros_like(f)
        corr_s = np.zeros_like(f)
        p_s = np.zeros_like(f)        
        for index in range(len(f)):
            pearson = stats.pearsonr(
                spec_dict_result['North_Spectrogram_Time'] ,
                spec_dict_result['North_Spectrogram'][index] )            
            corr_n[index] = pearson.statistic
            p_n[index] =    pearson.pvalue
            pearson = stats.pearsonr(
                spec_dict_result['South_Spectrogram_Time'] ,
                spec_dict_result['South_Spectrogram'][index] )            
            corr_s[index] = pearson.statistic
            p_s[index] =    pearson.pvalue
            
        temp['North_v_time_r'] = corr_n
        temp['North_v_time_p'] = p_n
        temp['South_v_time_r'] = corr_s
        temp['South_v_time_p'] = p_s
    
    
        # 20logr
        # Check for r < and > of spec length
        # Linear values cast to dB for this transform only.
        dr = r[1] - r[0]
        while (len(r) < spec_dict_result['North_Spectrogram'].shape[1]):
            r = list(r)
            r.append(r[-1] + dr)
        while (len(r) > spec_dict_result['North_Spectrogram'].shape[1]):
            r.pop(-1)
        corr_n = np.zeros_like(f)
        p_n = np.zeros_like(f)
        corr_s = np.zeros_like(f)
        p_s = np.zeros_like(f)        
        for index in range(len(f)):
            pearson = stats.pearsonr(
                20*np.log10(r) ,
                10*np.log10(spec_dict_result['North_Spectrogram'][index] ) )           
            corr_n[index] = pearson.statistic
            p_n[index] =    pearson.pvalue
            pearson = stats.pearsonr(
                20*np.log10(r) ,
                10*np.log10(spec_dict_result['South_Spectrogram'][index] )  )      
            corr_s[index] = pearson.statistic
            p_s[index] =    pearson.pvalue

        temp['North_v_20log_r'] = corr_n
        temp['North_v_20log_p'] = p_n
        temp['South_v_20log_r'] = corr_s
        temp['South_v_20log_p'] = p_s            

        # south hydro X v north hydro Y
        corr = np.zeros_like(f)
        p = np.zeros_like(f)        
        for index in range(len(f)):
            pearson = stats.pearsonr(
                spec_dict_result['South_Spectrogram'][index] ,
                spec_dict_result['North_Spectrogram'] [index])                
            corr[index] = pearson.statistic
            p[index] =    pearson.pvalue
        temp['North_v_South_r'] = corr
        temp['North_v_South_p'] = p
        results_dict[run] = temp

    results_dict['Frequency'] = f

    return results_dict
    

def plot_corr_regress_results(p_results_dict,
                             p_runID,
                             p_keys=['North_v_time_slope',
                                     'South_v_time_slope',
                                     'North_v_South_r'],
                             p_f_high = 10000):
    L = p_keys[0]
    C = p_keys[1]
    R = p_keys[2]
    
    t = p_results_dict[p_runID]
    f = p_results_dict['Frequency']
    
    
    
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(12,6))
    freqs = p_results_dict['Frequency']
    index_high = np.where(freqs > p_f_high)[0][0]
    axs[0].plot(f[:index_high],t[L][:index_high]);plt.xscale('log') 
    axs[0].set_xscale('log')
    axs[0].set_title(L)
    
    axs[1].plot(f[:index_high],t[C][:index_high]);plt.xscale('log') 
    axs[1].set_xscale('log')
    axs[1].set_title(C)
    
    axs[2].plot(f[:index_high],t[R][:index_high]);plt.xscale('log') 
    axs[2].set_xscale('log')
    axs[2].set_title(R)
    
    fig.supxlabel('Frequency (Hz)')
    fig.supylabel('Statistic value')
    fig.suptitle(
        'Linear regression statistics for ' + p_runID + \
            '\n Total time: ' + str(p_results_dict[p_runID]['Total time (s)']) + ' seconds' )
    return fig,axs


def run_linear_regressions(p_do_all_runs):
    """
    Compute correlation and p-value for said correlation for
    each frequency spectrum line in 
    specs v time
    specs v 20logR
    specs v specs (n v s)    
    
    the regression with 20logR requires taking decibel of the values first
    or else it doesn't mean anything.
    """
    mgr, list_runs, config_dictionary = \
        Real_Data_Manager.get_manager_and_runs() #has default args defined.
    
    if p_do_all_runs == True: 
        list_runs = [x for x in mgr.list_runs if 'AM' not in x]
        
        # The returned list runs is different from teh stored list runs.
        # The returned list is based on the _variables.py entries.
    
    results_dict = dict()
    for run in list_runs:
        temp = dict()  # each run will have results put here
        spec_dict_result = \
            mgr.load_target_spectrogram_data(
                run, 
                _directories_and_files.DIR_SPECTROGRAM)
    
        f = spec_dict_result['Frequency']
        r = spec_dict_result['R']   # distance from CPA
        r = np.sqrt (r**2 + 100**2) # distance from hydrophone
      
        # time, a proxy for x-y.
        corr_n          = np.zeros_like(f)
        p_n             = np.zeros_like(f)
        slope_n         = np.zeros_like(f)
        slope_n_std     = np.zeros_like(f)
        corr_s          = np.zeros_like(f)
        p_s             = np.zeros_like(f)        
        slope_s         = np.zeros_like(f)
        slope_s_std     = np.zeros_like(f)
        for index in range(len(f)):
            pearson = stats.linregress(
                spec_dict_result['North_Spectrogram_Time'] ,
                spec_dict_result['North_Spectrogram'][index] )            
            corr_n[index]       = pearson.rvalue
            p_n[index]          = pearson.pvalue
            slope_n[index]      = pearson.slope
            slope_n_std[index]  = pearson.stderr
            
            pearson = stats.linregress(
                spec_dict_result['South_Spectrogram_Time'] ,
                spec_dict_result['South_Spectrogram'][index] )            
            corr_s[index]       = pearson.rvalue
            p_s[index]          = pearson.pvalue
            slope_s[index]      = pearson.slope
            slope_s_std[index]  = pearson.stderr
            
            
        temp['North_v_time_r'] = corr_n
        temp['North_v_time_p'] = p_n
        temp['North_v_time_slope'] = slope_n
        temp['North_v_time_slope_std'] = slope_n_std
        
        temp['South_v_time_r'] = corr_s
        temp['South_v_time_p'] = p_s
        temp['South_v_time_slope'] = slope_s
        temp['South_v_time_slope_std'] = slope_s_std
    
    
        # 20logr
        # Check for r < and > of spec length
        # This only makes sense as a metric in the dB domain!
        dr = r[1] - r[0]
        while (len(r) < spec_dict_result['North_Spectrogram'].shape[1]):
            r = list(r)
            r.append(r[-1] + dr)
        while (len(r) > spec_dict_result['North_Spectrogram'].shape[1]):
            r.pop(-1)
        corr_n          = np.zeros_like(f)
        p_n             = np.zeros_like(f)
        slope_n         = np.zeros_like(f)
        slope_n_std     = np.zeros_like(f)
        corr_s          = np.zeros_like(f)
        p_s             = np.zeros_like(f)        
        slope_s         = np.zeros_like(f)
        slope_s_std     = np.zeros_like(f)
             
        for index in range(len(f)):
            pearson = stats.linregress(
                20*np.log10(r) ,
                10*np.log10(spec_dict_result['North_Spectrogram'][index] ) )        
            corr_n[index]       = pearson.rvalue
            p_n[index]          = pearson.pvalue
            slope_n[index]      = pearson.slope
            slope_n_std[index]  = pearson.stderr
            
            
            pearson = stats.linregress(
                20*np.log10(r) ,
                10*np.log10(spec_dict_result['South_Spectrogram'][index] ) )
            corr_s[index]       = pearson.rvalue
            p_s[index]          = pearson.pvalue
            slope_s[index]      = pearson.slope
            slope_s_std[index]  = pearson.stderr
            
            
        temp['North_v_20log_r'] = corr_n
        temp['North_v_20log_p'] = p_n
        temp['North_v_20log_slope'] = slope_n
        temp['North_v_20log_slope_std'] = slope_n_std        
        
        temp['South_v_20log_r'] = corr_s
        temp['South_v_20log_p'] = p_s            
        temp['South_v_20log_slope'] = slope_s
        temp['South_v_20log_slope_std'] = slope_s_std

        # south hydro X v north hydro Y
        corr          = np.zeros_like(f)
        p             = np.zeros_like(f)
        slope         = np.zeros_like(f)
        slope_std     = np.zeros_like(f)
        for index in range(len(f)):
            pearson = stats.linregress(
                spec_dict_result['South_Spectrogram'][index] ,
                spec_dict_result['North_Spectrogram'] [index])                
            corr[index]          = pearson.rvalue
            p[index]             = pearson.pvalue
            slope_s [index]      = pearson.slope
            slope_s_std[index]   = pearson.stderr
        
        temp['North_v_South_r'] = corr
        temp['North_v_South_p'] = p
        temp['North_v_South_slope'] = slope
        temp['North_v_South_slope_std'] = slope_std
        temp['Total time (s)'] = \
            len(spec_dict_result['South_Spectrogram'][index]) / 10 #gps 

        results_dict[run] = temp

    results_dict['Frequency'] = f

    return results_dict