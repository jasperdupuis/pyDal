# -*- coding: utf-8 -*-
"""

a file for the Buckleys of code:
    It's ugly, but it's works
    
"""





keys = list(series.keys())
r = keys[0]
x = 10** (series[r]/10)
len(x)


si = mystats.scintillation_index(x)
s = mystats.calc_skew(x)
k = mystats.calc_kurtosis(x)
