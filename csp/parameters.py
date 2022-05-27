# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:43:24 2022

@author: Jasper
"""

FS = 25600
num_seconds_chunk = 3
num_seconds_total = 20
N_chunk = FS * num_seconds_chunk
N_total = FS * num_seconds_total #largest effect on del_alpha and max alpha
overlap_major = 0.75 # Overlap of chunks to make a single gram
overlap_minor = 0.75 # Overlap of windows used in making a single gram, often denoted L in integer form
del_f = 25 # a major determinant in how much cyclic resolution you get. Higher delf ==> more alpha.

ICMC_MIN = 50 #index
ICMC_MAX = 500 #index

run_ids = ['DRJ1PB03AX00EB',
'DRJ1PB05AX00EB',
'DRJ1PB07AX00WB',
'DRJ1PB09AX00WB',
'DRJ1PB11AX00WB',
'DRJ1PB13AX00WB',
'DRJ1PB15AX00WB',
'DRJ1PB17AX00WB']

hull_map_2019 = {'H1' : 'Port outboard', 
                'H2' : 'Port inboard',
                'H3' : 'Starboard outboard',
                'H4' : 'Starboard inboard'}

col_burnsi_id = 'Run ID'
col_tdms_file = 'Onboard TDMS file'
col_hydro_file = 'South hydrophone raw'

df_fname = r'C:/Users/Jasper/Desktop/MASC/raw_data/burnsi_files_RECONCILE_20201125.csv'
tdms_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\AllTDMS\\'
hydro_dir = r'C:\Users\Jasper\Desktop\MASC\raw_data\2019-Orca Ranging\Range Data Amalg\ES0451_MOOSE_OTH_DYN\RAW_TIME\\'
