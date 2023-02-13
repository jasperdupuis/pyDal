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
        self.mgr.load_summary_stats(self.dir_spec_data + p_fname)




