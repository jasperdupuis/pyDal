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
dictionary['emerald_basin'] = [10,25,50,500]


for key,list_freq in dictionary.items():
    for FREQ in list_freq:
        with open('results//' + key + r'/data/ferguson_'+str(FREQ)+'.txt', 'r') as f:
            x_ram = f.readline()
            y_ram = f.readline()
            x_krak = f.readline()
            y_krak = f.readline()
            x_bell = f.readline()
            y_bell = f.readline()
                
        
        # # Generate plot
        # plt.plot(x_bell,20*np.log10(np.abs(y_bell)),label='BELLHOP')
        # plt.plot(x_krak,y_krak,label='KRAKEN')
        # plt.plot(x_ram,y_ram,label='RAM')
        # plt.plot(x_bell,-20*np.log10(x_bell),label='20logR')
        # if key =='ferguson':
        #     plt.title('Ferguson Cove, ' + str(FREQ) + ' Hz')
        # if key =='emerald_basin':
        #     plt.title('Emerald Basin, ' + str(FREQ) + ' Hz')
        # if key =='pekeris':
        #     plt.title('Pekeris Waveguide, ' + str(FREQ) + ' Hz')
        # plt.ylabel('Transmission Loss (dB ref 1uPa^2)')
        # plt.xlabel('Distance (m)')
        # plt.ylim(-60,-20)
        # plt.legend()
        # plt.savefig( dpi = 300,
        #             fname = 'results//' + key + r'/png//' + key + r'_'+str(FREQ)+r'.png')
        # plt.savefig(dpi = 300,
        #             fname = 'results//' + key + r'/pdf//' + key + r'_'+str(FREQ)+r'.pdf')
        # plt.close('all')