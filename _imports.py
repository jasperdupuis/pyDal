# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:09:20 2023

@author: Jasper
"""

import os as os
import numpy as np
import h5py as h5
import pickle as pickle
from scipy import interpolate, signal, stats
import pandas as pd
import matplotlib.pyplot as plt

#Do I even use this anymore in this work? Answer : Yes.
import sys
sys.path.insert(1, r'C:\pydrdc')
import signatures
RANGE_DICTIONARY = \
    signatures.data.range_info.dynamic_patbay_2019.RANGE_DICTIONARY

# My global modules
from env.locations import Location 
from env.bathymetry import Bathymetry_CHS_2

from data_access.real_hydrophone import Real_Hydrophone as Real_Hyd
from data_access.real_ambients import Real_Ambient as Real_Amb
# The above two must be imported before the below one.
from data_access.real_accessor_class import Real_Data_Manager

from data_analysis.moment_analyzer import Moment_Analyzer

