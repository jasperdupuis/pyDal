# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:32:53 2022

@author: Jasper
"""





keys = list(series.keys())
r = keys[0]
x = 10** (series[r]/10)
len(x)


si = mystats.scintillation_index(x)
s = mystats.calc_skew(x)
k = mystats.calc_kurtosis(x)
