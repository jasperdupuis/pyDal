# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 13:38:09 2021

@author: Jasper
"""

import numpy as np
import matplotlib.pyplot as plt

import haversine
import arlpy.uwapm as pm

#my modules
from env.environment import Environment_ARL,Environment_PYAT
from env.bathymetry import Bathymetry_WOD
from env.ssp import SSP_Blouin_2015
from env.seabed import SeaBed
from env.surface import Surface

import source

LOCATION = 'NS'
# LOCATION = 'BC'

BASIS_SIZE = 20
N_BEAMS = 100
FREQ = 20000

RX_HYD_DEPTH = 18.9 #m

IR_RESULTS = []
TL_RESULTS = []
HYD_1_R = []
HYD_2_R = []

IMPULSE_RESPONSE_REL_TX_TIME = True

if LOCATION == 'NS':
    # NS COORDINATES AND FILES
    hyd_1_lat   = 44.60551039
    hyd_1_lon   = -63.54031211
    hyd_2_lat   = 44.60516000
    hyd_2_lon   = -63.54272550    
    fname_bathy_NS = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
    fname_3rd_order_coef = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
    bottom_id  ='Silty-clay'

CPA_lat = ( hyd_1_lat + hyd_2_lat ) / 2
CPA_lon = ( hyd_1_lon + hyd_2_lon ) / 2


#for reducing the large WOD data to a small window on range
if hyd_1_lat > hyd_2_lat: #hyd 1 is more north than hyd 2
    lat_extent_tuple = (hyd_2_lat-0.01,hyd_1_lat+0.01)
else: lat_extent_tuple = (hyd_1_lat-0.01,hyd_2_lat+0.01)
if hyd_1_lon > hyd_2_lon: #hyd 1 is more east than hyd 2
    lon_extent_tuple =  (hyd_2_lon-0.01,hyd_1_lon+0.01)
else: lon_extent_tuple =  (hyd_1_lon-0.01,hyd_2_lon+0.01)

ssp = SSP_Blouin_2015()
ssp.set_depths(np.arange(0,50,5))
ssp.read_profile(fname_3rd_order_coef)

bathymetry = Bathymetry_WOD()
bathymetry.read_bathy(fname_bathy_NS)
bathymetry.sub_select_by_latlon(
    p_lat_extent_tuple = lat_extent_tuple,
    p_lon_extent_tuple = lon_extent_tuple) #has default values for NS already
bathymetry.interpolate_bathy()

bottom_profile = SeaBed(bathymetry)
bottom_profile.read_default_dictionary()
bottom_profile.assign_single_bottom_type(bottom_id)

surface = Surface()

THE_ENVIRONMENT = Environment_ARL()
THE_ENVIRONMENT.set_bathymetry(bathymetry)
THE_ENVIRONMENT.set_seabed(bottom_profile)
THE_ENVIRONMENT.set_ssp(ssp)
THE_ENVIRONMENT.set_surface(surface)

THE_SOURCE = source.Source()
THE_SOURCE.set_name()
THE_SOURCE.set_depth()
THE_SOURCE.set_speed()
THE_SOURCE.generate_course(((CPA_lat),(CPA_lon)),
                        p_CPA_deviation_m = 5,
                        p_CPA_deviation_heading=haversine.Direction.EAST,
                        p_course_heading=1.99*np.pi, #(0,2pi), mathematical not navigation angles
                        p_distance=200,
                        p_divisions = 25)

# DISPLAY ENVIRONMENTAL PARAMETERS  (Optional)
# pm.print_env(env)

#for FREQ in FREQS_TO_RUN:
for TX_SOURCE in THE_SOURCE.course:
    env = THE_ENVIRONMENT.create_environment_model(
            (hyd_1_lat,hyd_1_lon),
            TX_SOURCE,
            FREQ_TO_RUN = FREQ,
            RX_HYD_DEPTH = RX_HYD_DEPTH,
            TX_DEPTH = THE_SOURCE.depth,
            N_BEAMS = N_BEAMS,
            BASIS_SIZE = BASIS_SIZE
            )
    
    # APPLY FERG RANGE DEPTH CORRECTION FROM BATHY DATA.
    # Due to coastal geography interpolation function doesn't go deep enough.
    # REMOVE THIS FOR COMPARISON'S SAKE,
    # MUST BE DONE IN GENERAL FASHION FOR ALL MODEL TYPES
    
    
    rays = pm.compute_eigenrays(env)
    # pm.plot_rays(rays, env=env, width=900) # ONLY WORKS IN JUPYTER
    # compute the arrival structure at the receiver
    
    arrivals = pm.compute_arrivals(env)
    #pm.plot_arrivals(arrivals, width=900) # ONLY WORKS IN JUPYTER
    
    ir = pm.arrivals_to_impulse_response(
        arrivals,
        fs=204800,
        abs_time = IMPULSE_RESPONSE_REL_TX_TIME
        )
    IR_RESULTS.append(ir)
    # plt.plot(np.abs(ir),label=str(FREQ_TO_RUN))
    # plt.legend()

    #ac_plt.plot(np.abs(ir), fs=96000, width=900) #ONLY WORKS IN JUPYTER
    
    TL = pm.compute_transmission_loss(
        env,
        mode=pm.coherent,
        )
    x_cmplx = TL.iloc(0)[0].iloc(0)[0]
    TL_RESULTS.append(x_cmplx)
    HYD_1_R.append(env['rx_range'])
    # loss = 20*np.log10(np.abs(x_cmplx))

# x = np.array(IR_RESULTS)
# x = np.sum(x,axis=0)
# plt.plot((np.abs(x[1])*50) -  np.abs(np.sum(x,axis=0)))

# np.sum(np.abs(x[49]-x[40]))

y = np.array(TL_RESULTS)
#plt.plot(FREQS_TO_RUN,20*np.log10(np.abs(y)))

plt.plot(HYD_1_R,20*np.log10(np.abs(y)))
plt.plot(HYD_1_R,-20*np.log10(HYD_1_R))

arr = np.array(THE_SOURCE.course)
lats = arr[:,0]
lons = arr[:,1]
plt.plot(lons,lats,label='Source course')
plt.plot(hyd_1_lon,hyd_1_lat,marker='X',color='r',label='Hyd 1')
plt.plot(hyd_2_lon,hyd_2_lat,marker='X',color='g',label='Hyd 2')
plt.legend()
