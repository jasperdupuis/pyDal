# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:26:17 2021

KRAKEN specific functions

@author: Jasper
"""

from os import system
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')

import pyat.pyat.env
import pyat.pyat.readwrite


def calculate_modes_and_pressure_field(env_pyat,
                                        pos):
    """
    wrap the KRAKEN mode calculation
    
    returns 20log(pressure), i.e. intensity
    """    
    pyat.pyat.readwrite.write_env(
        'py_env.env',
        'KRAKEN',
        'Pekeris profile',
        env_pyat.freq,
        env_pyat.ssp_pyat,
        env_pyat.bdy,
        pos,
        env_pyat.beam,
        env_pyat.cInt,
        max(env_pyat.X))
    
    
    pyat.pyat.readwrite.write_fieldflp('py_env', 'R', pos)
    system("krakenc.exe py_env")
    fname = 'py_env.mod'
    options = {'fname':fname, 'freq':0}
    modes = pyat.pyat.readwrite.read_modes(**options)
    delta_k = np.max(modes.k.real) - np.min(modes.k.real)
    bandwidth = delta_k * 2.5 / 2 / (2*np.pi)
    system("field.exe py_env")
    [x,x,x,x,Pos1,pressure]= pyat.pyat.readwrite.read_shd('py_env.shd')
    pressure = abs(pressure)
    pressure = 20*np.log10(pressure) #returns intensity
    return modes, bandwidth, Pos1,pressure


# def calculate_modes_and_pressure_field(freq, 
#                                        ssp, 
#                                        bdy, 
#                                        pos, 
#                                        cInt,
#                                        RMax,
#                                        arry=[]):
#     """
#     wrap the KRAKEN mode calculation
#     """    
#     pyat.pyat.readwrite.write_env(
#         'py_env.env',
#         'KRAKEN',
#         'Pekeris profile',
#         freq, 
#         ssp, 
#         bdy, 
#         pos, 
#         arry, 
#         cInt,
#         RMax)
 
#     pyat.pyat.readwrite.write_fieldflp('py_env', 'R', pos)
#     system("krakenc.exe py_env")
#     fname = 'py_env.mod'
#     options = {'fname':fname, 'freq':0}
#     modes = pyat.pyat.readwrite.read_modes(**options)
#     delta_k = np.max(modes.k.real) - np.min(modes.k.real)
#     bandwidth = delta_k * 2.5 / 2 / (2*np.pi)
#     system("field.exe py_env")
#     [x,x,x,x,Pos1,pressure]= pyat.pyat.readwrite.read_shd('py_env.shd')
#     pressure = abs(pressure)
#     pressure = 10*np.log10(pressure / np.max(pressure))
#     levs = np.linspace(np.min(pressure), np.max(pressure), 20)

#     return modes, bandwidth, Pos1,pressure,levs

def plot_and_show_result(Pos1,pressure):
    levs = np.linspace(np.min(pressure), np.max(pressure), 20)
    plt.contourf(Pos1.r.range, Pos1.r.depth,(pressure[0, 0,:,:]),levels=levs)
    plt.gca().invert_yaxis()
    plt.show()
    plt.colorbar()
    return