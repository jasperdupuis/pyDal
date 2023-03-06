# -*- coding: utf-8 -*-
"""
Script to use the functions across this package


@author: Jasper
"""

import time

import sys
sys.path.append('C:\\Users\\Jasper\\Documents\\Repo\\pyDal\\pyDal\\env')
import os

import _imports
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


if __name__ == '__main__':
    
    start = time.time() #for all your timing needs.
    # Off the shelf useful scripts, just uncomment them.
    # from scripts import analysis_main # This does something useful.
    # from env import create_TL_models  # This does something useful.
    
    # Correlation uses several functions for exploration.
    from scripts.analysis_correlations import \
        run_linear_regressions,\
        plot_corr_regress_results
        
    results_linregress = run_linear_regressions(p_do_all_runs = True)
    list_linreg_runs = list(results_linregress.keys())
    runID = list_linreg_runs[1] # user selected for testing now.
    list_linreg_strings = list(results_linregress[runID].keys())
    plot_strings = [list_linreg_strings[1],
                    list_linreg_strings[5],
                    list_linreg_strings[16]]
    plot_strings_secondary = [list_linreg_strings[3],
                    list_linreg_strings[7],
                    list_linreg_strings[16]]

    # f,a = plot_corr_regress_results(
    #     results_linregress,
    #     p_runID     = runID,
    #     p_keys      =plot_strings,
    #     p_f_high    = 10000)

    # f2,a2 = plot_corr_regress_results(
    #     results_linregress,
    #     p_runID     = runID,
    #     p_keys      = plot_strings_secondary,
    #     p_f_high    = 10000)


    end = time.time()
    print ('elapsed time: ' + str(end-start)[:4] + ' seconds')
    
    