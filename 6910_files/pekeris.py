"""
This is the Buckleys of code.

It looks awful, but it works.

IN particular, making this an adaptive, input-based code required
jury rigging depths and depth index for the RX depth.



NOTE this has MOVING SOURCE only right now.
"""

import numpy as np
import matplotlib.pyplot as plt

import haversine
import arlpy.uwapm as pm

import sys
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')
import pyat

#my modules
from env.environment import create_basis_common,Environment_ARL,Environment_PYAT,Environment_RAM
from env.bathymetry import Bathymetry_WOD
from env.ssp import SSP_Blouin_2015, SSP_Munk, SSP_Isovelocity
from env.seabed import SeaBed
from env.surface import Surface
from env.locations import Location
import env.comparison_setup

import source
import KRAKEN_functions

LOCATION = 'Pekeris Waveguide'
the_location = Location(LOCATION) 

# Parameters to manipulate
FREQS = [10,12.5,16,20,25,31.5,40,50,63,80,100,125,160,200,250,315,400,500]
RX_HYD_DEPTH = 46 # m
TX_DEPTH = 36
COURSE_heading = 1.99 #heading, cartesian not navigational angle.
num_COURSE_POINTS = 250

#Don't usually change these, but can.
BASIS_SIZE_depth = 5000
BASIS_SIZE_distance = 5000
PEKERIS_DEPTH = 100
N_BEAMS = 100 #this doesn't appear to matter - bellhop parameter
KRAKEN_ROUGHNESS = [0.1,0.1] # Doesn't change anything apparently
RAM_DELTA_R = 1 # m, range step size
RAM_DELTA_Z = 1 # m, depth step size, not currently used by me (Default is computed in pyram)
    
THE_SOURCE, env_ARL, env_PYAT, env_RAM = env.comparison_setup.setup(
        the_location,
        p_source_depth = TX_DEPTH,
          p_course_heading = COURSE_heading,
          p_course_num_points = num_COURSE_POINTS,
          p_pekeris_depths = PEKERIS_DEPTH,
          p_basis_size_depth = BASIS_SIZE_depth,
          p_basis_size_range = BASIS_SIZE_distance,
          p_kraken_roughness = KRAKEN_ROUGHNESS,
          p_ram_delta_r = RAM_DELTA_R,
          p_depth_offset = 0,
          p_CPA_offset = 0
          )

#If needed
lat_end = THE_SOURCE.course[-1][0]
lon_end = THE_SOURCE.course[-1][1]
lat_start = THE_SOURCE.course[0][0]
lon_start = THE_SOURCE.course[0][1]

for FREQ in FREQS:
    #
    #
    #
    # BELLHOP - compute every source change.    
    #Results storage
    TL_RES_BELL = []
    HYD_1_R = []
    HYD_2_R = []
    count = 0
    for TX_SOURCE in THE_SOURCE.course:
        if TX_SOURCE[0] == lat_end: break
        # BELLHOP first, then KRAKEN afterwards
        env_bellhop = env_ARL.create_environment_model(
                (lat_end,lon_end),
                TX_SOURCE,
                FREQ_TO_RUN = FREQ,
                RX_HYD_DEPTH = RX_HYD_DEPTH,
                TX_DEPTH = THE_SOURCE.depth,
                N_BEAMS = N_BEAMS,
                BASIS_SIZE_DEPTH = BASIS_SIZE_depth,
                BASIS_SIZE_DISTANCE = BASIS_SIZE_distance,
            )
        TL = pm.compute_transmission_loss(
            env_bellhop,
            mode=pm.coherent,
            )
        x_cmplx = TL.iloc(0)[0].iloc(0)[0]
        TL_RES_BELL.append(x_cmplx)
        HYD_1_R.append(env_bellhop['rx_range'])
        print(str(count) +': '+ str(TX_SOURCE))
        count+=1
    
    #
    #
    #
    # KRAKEN
    # Compute Kraken, doesn't need to recompute for each receiver location!
    env_kraken = env_PYAT.create_environment_model(
            (lat_end,lon_end), #rx lat lon, rx depth already set 
            THE_SOURCE.course[0], #tx lat lon
            THE_SOURCE, # includes tx depth
            p_beam = [],
            BASIS_SIZE_depth = BASIS_SIZE_depth,
            BASIS_SIZE_distance = BASIS_SIZE_distance,
            freq = FREQ
            )
    
    modes, bandwidth, Pos1,pressure, rx_depths_index = \
        KRAKEN_functions.calculate_modes_and_pressure_field(
            env_kraken,
            SOURCE_DEPTH = THE_SOURCE.depth,
            BASIS_SIZE_distance = BASIS_SIZE_distance,
            BASIS_SIZE_depth = BASIS_SIZE_depth,
            RX_HYD_DEPTH = RX_HYD_DEPTH)
    
    #
    #
    #
    # RAM
    # Also does not need to be recomputed for each receiver location.
    env_RAM.create_environment_model(
                (lat_end,lon_end),
                (lat_start,lon_start),
                TX_DEPTH = THE_SOURCE.depth,
                FREQ_TO_RUN = FREQ,
                RX_DEPTH = RX_HYD_DEPTH,
                BASIS_SIZE_DEPTH = BASIS_SIZE_depth,
                BASIS_SIZE_DISTANCE = BASIS_SIZE_distance,
            )
    results_RAM = env_RAM.run_model()
    
    
    # Generate data    
    TL_RES_RAM = results_RAM['TL Line']
    y_ram = -1*np.array(TL_RES_RAM)
    x_ram = results_RAM['Ranges']
    
    y_bell = np.array(TL_RES_BELL)
    x_bell = np.array(HYD_1_R)
    
    y_krak = np.array(pressure[0,0,rx_depths_index,:])
    x_krak = Pos1.r.range
    
    
    # Save data
    with open('results/pekeris/data/pekeris_'+str(FREQ)+'.txt', 'w') as f:
        f.write('X_RAM (m):')
        for entry in x_ram: f.write(str(entry)+',')
        f.write('\nY_RAM (dB):')
        for entry in y_ram: f.write(str(entry)+',')
        f.write('\nX_KRAK (m):')
        for entry in x_krak: f.write(str(entry)+',')
        f.write('\nY_KRAK (dB):')
        for entry in y_krak: f.write(str(entry)+',')
        f.write('\nX_BELL (m):')
        for entry in x_bell: f.write(str(entry)+',')
        f.write('\nY_BELL (dB):')
        for entry in y_bell: f.write(str(entry)+',')
    
    
    # Generate plot
    plt.plot(x_bell,20*np.log10(np.abs(y_bell)),label='BELLHOP')
    plt.plot(x_krak,y_krak,label='KRAKEN')
    plt.plot(x_ram,y_ram,label='RAM')
    plt.plot(x_bell,-20*np.log10(x_bell),label='20logR')
    plt.title('Pekeris, ' + str(FREQ) + ' Hz')
    plt.ylabel('Transmission Loss (dB ref 1uPa^2)')
    plt.xlabel('Distance (m)')
    plt.legend()
    plt.savefig( dpi = 300,
                fname = 'results/pekeris/png/pekeris_'+str(FREQ)+'.png')
    plt.savefig(dpi = 300,
                fname = 'results/pekeris/pdf/pekeris_'+str(FREQ)+'.pdf')
    plt.close('all')