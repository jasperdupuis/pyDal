# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 10:09:16 2022

@author: Jasper
"""

import matplotlib.pyplot as plt
import numpy as np

dictionary = dict()
dictionary['ferguson'] = [40,100,200,500]
dictionary['pekeris'] = [10,25,50,500]
dictionary['emerald'] = [10,25,50,500]
key = 'pekeris'
list_freq = dictionary[key]

for key,list_freq in dictionary.items():
    for FREQ in list_freq:
        print(key + ': ' + str(FREQ))
        with open('results//' + key + r'/data//' + key + '_'+str(FREQ)+'.txt', 'r') as f:
            x_ram = f.readline().split(':')[1][:-2]
            y_ram = f.readline().split(':')[1][:-2]
            x_krak = f.readline().split(':')[1][:-2]
            y_krak = f.readline().split(':')[1][:-2]
            x_bell = f.readline().split(':')[1][:-2]
            y_bell = f.readline().split(':')[1][:-1]
            
            #turn in to arrays
            x_ram = np.array(x_ram.split(','),dtype=float)
            y_ram = np.array(y_ram.split(','),dtype=float)
            x_krak = np.array(x_krak.split(','),dtype=float)
            y_krak = np.array(y_krak.split(','),dtype=float)
            x_bell = np.array(x_bell.split(','),dtype=float)
            y_bell = np.array(y_bell.split(','),dtype=complex)

            # Generate plot
            plt.plot(x_bell,-20*np.log10(np.abs(y_bell)),label='BELLHOP')
            plt.plot(x_krak,y_krak,label='KRAKEN')
            plt.plot(x_ram,y_ram,label='RAM')
            plt.plot(x_bell,-20*np.log10(x_bell),label='20logR')
            if key =='ferguson':
                title = 'Ferguson Cove, ' + str(FREQ) + ' Hz'
            if key =='emerald_basin':
                title = 'Emerald Basin, ' + str(FREQ) + ' Hz'
            if key =='pekeris':
                title = 'Pekeris Waveguide, ' + str(FREQ) + ' Hz'
            plt.title(title)
            plt.ylabel('Transmission Loss (dB ref 1uPa^2)')
            plt.xlabel('Distance (m)')
            plt.ylim(-100,-20)
            plt.grid(axis='y',which='major')
            plt.legend()
            plt.savefig( dpi = 300,
                        fname = 'results//' + key + r'/png//' + key + r'_'+str(FREQ)+r'.png')
            plt.savefig(dpi = 300,
                        fname = 'results//' + key + r'/pdf//' + key + r'_'+str(FREQ)+r'.pdf')
            plt.close('all')
