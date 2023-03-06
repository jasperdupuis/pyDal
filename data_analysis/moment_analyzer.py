# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:00:28 2023

@author: Jasper
"""

from _imports import os, np, plt, Location, Real_Data_Manager

from _directories_and_files import \
    DIR_SPECTROGRAM,\
    SUMMARY_FNAME
    
class Moment_Analyzer():
    
    def __init__(
        self,
        p_dir_spectrogram = DIR_SPECTROGRAM):

        self.dir_spec_data = p_dir_spectrogram
        list_files = os.listdir(self.dir_spec_data )
        list_files = [x for x in list_files if 'summary' not in x]
        list_runs = [x.split('_')[0] for x in list_files]
        self.mgr = Real_Data_Manager(list_runs,self.dir_spec_data )    

        return 
    

    def load_summary_stats(self,p_fname = SUMMARY_FNAME):
        self.mgr.load_and_set_pickle_summary_stats(self.dir_spec_data + p_fname)


    def flatten_moments_ambient(self):
        """
        Summarize the ambient summary statistics into a single dimension
        """
        keys_all = self.mgr.summary_stats.keys() 
        keys_amb = [x for x in keys_all if 'AMJ' in x]
        key_any = keys_amb[1]

        # Find number of ambient runs to use
        n = 0
        for key in keys_amb:
            n = n + 1

        # make a results dictionary that makes an array for each key type
        result = dict()
        for k,v in self.mgr.summary_stats[key_any].items():
            if 'Freq' in k: continue
            result[k] = np.zeros([len(v),n])

        #now insert the values in to array
        index = 0
        for k,v in self.mgr.summary_stats.items(): # for each run in the summary stats dictionary
            if not(k in keys_amb): continue #only want ambients
            for kk,vv in self.mgr.summary_stats[k].items(): # each summary stat type in current run
                if 'Freq' in kk: continue
                result[kk][:,index] = vv
            index = index + 1
                          
        freq = self.mgr.summary_stats[key]['Frequency']
        self.dict_means = dict()
        self.dict_stds = dict()
        self.dict_means['Frequency'] = freq
        self.dict_stds['Frequency'] = freq
        for k,v in result.items():
            if 'freq' in k: continue
            mean = np.mean(v,axis = 1)
            std = np.std(v,axis=1)
            self.dict_means[k] = mean
            self.dict_stds[k] = std
        return self.dict_means, self.dict_stds
    
    def plot_all_ambient_moment_by_string(self,
                                             p_type = 'Skew',
                                             p_f_high=1000):
        keys_all = self.mgr.summary_stats.keys() 
        keys_amb = [x for x in keys_all if 'AMJ' in x]
        key_any = keys_amb[1]
        for k,v in self.mgr.summary_stats[key_any].items():
            if 'Freq' in k: 
                freqs = v
                break
            
        index_high = np.where(freqs > p_f_high)[0][0]
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
        #now insert the values in to array
        for k,v in self.mgr.summary_stats.items(): # for each run in the summary stats dictionary
            if not(k in keys_amb): continue #only want ambients
            for kk,vv in self.mgr.summary_stats[k].items(): # each summary stat type in current run
                if not(p_type in kk): continue # only plot the funciton argument
                ax.plot(freqs[:index_high],vv[:index_high])
                ax.set_xscale('log')
        fig.suptitle('All ambient ' +p_type+ ' data' )
        fig.supylabel(p_type)
        fig.supxlabel('Frequency (Hz)')
        return fig,ax
    
    def plot_ambient_results_return_fig_axs(self,p_f_high=10000):
        fig_mean, axs_mean = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
        freqs = self.dict_means['Frequency']
        index_high = np.where(freqs > p_f_high)[0][0]
        for k,v in self.dict_means.items(): # do north top south bottom
            if 'Frequency' in k : continue
            if 'Mean' in k :    continue
            if 'STD' in k :     continue
            
            if 'North' in k:    row = 0
            if 'South' in k:    row = 1 # South
            
            if 'Skew' in k:     col=0
            if 'Kurtosis' in k: col = 1
            if 'Scint' in k:    col = 2 #Scintiliation index
            axs_mean[row,col].plot(freqs[:index_high],v[:index_high])
            axs_mean[row,col].set_xscale('log')
            axs_mean[row,col].set_title(k)
        fig_mean.supylabel('Mean of linear moments')
   
        fig_std, axs_std = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
        for k,v in self.dict_stds.items(): # do north top south bottom
            if 'Frequency' in k : continue
            if 'Mean' in k :    continue
            if 'STD' in k :     continue
            
            if 'North' in k:    row = 0
            if 'South' in k:    row = 1 # South
            
            if 'Skew' in k:     col=0
            if 'Kurtosis' in k: col = 1
            if 'Scint' in k:    col = 2 #Scintiliation index
            axs_std[row,col].plot(freqs[:index_high],v[:index_high])
            axs_std[row,col].set_xscale('log')
            axs_std[row,col].set_title(k)
        fig_std.supylabel('Deviation of linear moments')
        fig_std.supxlabel('Frequency (Hz)')
        

        return fig_mean, axs_mean, fig_std, axs_std
         
    def plot_runID_and_ambient_mean_moments(self,p_runID,p_f_high=1000):
        """
        Plot the provided runID over the mean moments of ambients
        """
        fig_mean, axs_mean, fig_std , axs_std = \
            self.plot_ambient_results_return_fig_axs(p_f_high)
        plt.close(fig_std) #this func does means only

        freqs = self.mgr.summary_stats[p_runID]['Frequency']
        index_high = np.where(freqs > p_f_high)[0][0]
        for k,v in self.mgr.summary_stats[p_runID].items(): # do north top south bottom
            if 'Frequency' in k : continue
            if 'Mean' in k :    continue
            if 'STD' in k :     continue
            
            if 'North' in k:    row = 0
            if 'South' in k:    row = 1 # South
            
            if 'Skew' in k:     col=0
            if 'Kurtosis' in k: col = 1
            if 'Scint' in k:    col = 2 #Scintiliation index
            axs_mean[row,col].plot(freqs[:index_high],v[:index_high])
            axs_mean[row,col].set_xscale('log')
            axs_mean[row,col].set_title(k)        
        fig_mean.suptitle('Comparison of ' + p_runID +' moments to ambient averages')
        
        return fig_mean,axs_mean
    
    
    def plot_all_runs_and_ambients_with_fmax(self,
                                             p_fpath = r'C:\Users\Jasper\Documents\Repo\Results (offline)\Runs and ambient moments summary\to 1khz\\',
                                             p_f_h = 1000):
        keys = self.mgr.summary_stats.keys()
        runs = [x for x in keys if 'DRJ' in x]
        for runID in runs:
            fname = p_fpath + runID        
            fig, axs = self.plot_runID_and_ambient_mean_moments(runID,p_f_h)
            plt.savefig(dpi = 300, fname = fname + '.png')
            plt.savefig(dpi = 300, fname = fname + '.pdf')
            plt.close('all')

    def plot_all_ambient_spectra(self):
        fig_n, ax_n, fig_s, ax_s = \
            self.mgr.plot_all_ambients(p_f_high=1000)
        return fig_n, ax_n, fig_s, ax_s 




if __name__ == '__main__':
    import sys
    sys.path.insert(1, r'C:\Users\Jasper\Documents\Repo\pyDal\pyDal')
    # LINEAR STATS
    # fname = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal/data_access/real_data_sets/hdf5_timeseries_bw_01_overlap_90/summary_stats_dict.pkl'
    # DECIBEL STATS
    fname = r'C:/Users/Jasper/Documents/Repo/pyDal/pyDal/data_access/real_data_sets/hdf5_timeseries_bw_01_overlap_90/summary_stats_dict_DECIBEL.pkl'
    analyzer = Moment_Analyzer()
    _ = analyzer.mgr.load_and_set_pickle_summary_stats(fname)

    means, std = analyzer.flatten_moments_ambient()
    fpath = r'C:\Users\Jasper\Documents\Repo\Results (offline)\Runs and ambient moments summary\to 1khz decibels\\'
    f_h = 1000
    # analyzer.plot_all_runs_and_ambients_with_fmax(fpath,f_h)

    fig,ax = analyzer.plot_all_ambient_moment_by_string('Kurtosis')
    
    fig_n,ax_n,fig_s,ax_s = analyzer.plot_all_ambient_spectra()

