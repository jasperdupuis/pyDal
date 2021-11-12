# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:41:42 2021

@author: Jasper
"""

import matplotlib.pyplot as plt

import numpy as np
import ast

fname_3rd_order_coef = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'

with open(fname_3rd_order_coef) as f:
    data = f.readlines()

profiles_dict = dict()
for line in data:
    strs = line.split('[')
    coefs = ast.literal_eval('['+strs[1])
    strs[0].split(' ')[0]
    profiles_dict[strs[0].split(' ')[0]] = coefs

