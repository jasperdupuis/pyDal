"""
This is the Buckleys of code.

It looks awful, but it works.

IN particular, making this an adaptive, input-based code required
jury rigging depths and depth index for the RX depth.
"""

import numpy as np
import matplotlib.pyplot as plt

import haversine
import arlpy.uwapm as pm

import sys
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')
import pyat

#my modules
from env.environment import create_basis_common,Environment_ARL,Environment_PYAT
from env.bathymetry import Bathymetry_WOD
from env.ssp import SSP_Blouin_2015, SSP_Munk, SSP_Isovelocity
from env.seabed import SeaBed
from env.surface import Surface
from env.locations import Location

import source
import KRAKEN_functions

DEG_OFFSET = 0.75 # Each location has a centre point, this defines the decimal degrees to each side from that.
# LOCATION = 'Emerald Basin'
# LOCATION = 'Pat Bay'
# LOCATION = 'Ferguson Cove'
LOCATION = 'Pekeris Waveguide'
the_location = Location(LOCATION,DEG_OFFSET)
    

# Parameters to manipulate
FREQ = 500
RX_HYD_DEPTH = 50 #m
BASIS_SIZE_depth = 5000
BASIS_SIZE_distance = 500
N_BEAMS = 100 #this doesn't appear to matter - bellhop parameter
PEKERIS_DEPTH = 100

#Results storage
TL_RES_BELL = []
TL_RES_KRAK = []
HYD_1_R = []
HYD_2_R = []
KRAKEN_ROUGHNESS = [0.1,0.1] # Doesn't change anything apparently
    

#Set bathymetry
bathymetry = Bathymetry_WOD()
bathymetry.read_bathy(the_location.fname_bathy)
bathymetry.sub_select_by_latlon(
    p_lat_extent_tuple = the_location.LAT_EXTENT_TUPLE,
    p_lon_extent_tuple = the_location.LON_EXTENT_TUPLE) #has default values for NS already
bathymetry.interpolate_bathy()

if the_location.location_title == 'Pekeris Waveguide':
    #set the constant depth
    bathymetry.z = PEKERIS_DEPTH

#set source properties
THE_SOURCE = source.Source()
THE_SOURCE.set_name()
THE_SOURCE.set_depth() #Default is 4m
THE_SOURCE.set_speed()
THE_SOURCE.generate_course((the_location.LAT,the_location.LON),
                        p_CPA_deviation_m = 0,
                        p_CPA_deviation_heading=haversine.Direction.EAST,
                        p_course_heading=1.99*np.pi, #(0,2pi), mathematical not navigation angles
                        p_distance=5000, # m
                        p_divisions = BASIS_SIZE_distance) #number of divisions

#get the 2D plane of the course
total_distance,distances, z_interped, depths = \
    create_basis_common(
        bathymetry,
        THE_SOURCE.course[0],
        THE_SOURCE.course[-1],
        BASIS_SIZE_depth,
        BASIS_SIZE_distance)

# THIS IS KIND OF OUT OF ORDER, BUT NEED IT HERE.
MAX_LOCAL_DEPTH = np.abs(np.min(z_interped))

#ssp = SSP_Blouin_2015()
ssp = SSP_Munk()
ssp.set_depths(np.linspace(0,MAX_LOCAL_DEPTH,BASIS_SIZE_depth))
ssp.read_profile(the_location.ssp_file)
# ssp = SSP_Isovelocity()
#ssp.set_ssp(1485)

bottom_profile = SeaBed(bathymetry)
bottom_profile.read_default_dictionary()
bottom_profile.assign_single_bottom_type(the_location.bottom_id)

surface = Surface()

env_ARL = Environment_ARL()
env_ARL.set_bathymetry(bathymetry)
env_ARL.set_seabed(bottom_profile)
env_ARL.set_ssp(ssp)
env_ARL.set_surface(surface)

env_PYAT = Environment_PYAT()
env_PYAT.set_bathymetry(bathymetry)
env_PYAT.set_seabed(bottom_profile)
env_PYAT.set_ssp(ssp,KRAKEN_ROUGHNESS)


THE_SOURCE = source.Source()
THE_SOURCE.set_name()
THE_SOURCE.set_depth() #Default is 4m
THE_SOURCE.set_speed()
THE_SOURCE.generate_course((the_location.LAT,the_location.LON),
                        p_CPA_deviation_m = 0,
                        p_CPA_deviation_heading=haversine.Direction.EAST,
                        p_course_heading=1.99*np.pi, #(0,2pi), mathematical not navigation angles
                        p_distance=5000,
                        p_divisions = BASIS_SIZE_distance)

#Generate the Bellhop result which requires recomputing every move.
lat_end = THE_SOURCE.course[-1][0]
lon_end = THE_SOURCE.course[-1][1]
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

#Compute Kraken, which doesn't need to be reocmputed for each move!
# Now KRAKEN
env_kraken = env_PYAT.create_environment_model(
        (lat_end,lon_end), #rx lat lon, rx depth already set 
        THE_SOURCE.course[0], #tx lat lon
        THE_SOURCE, # includes tx depth
        p_beam = [],
        BASIS_SIZE_DEPTH = BASIS_SIZE_depth,
        BASIS_SIZE_DISTANCE = BASIS_SIZE_distance,
        freq = FREQ
        )

s = pyat.pyat.env.Source([THE_SOURCE.depth])
bottom_max = np.max(abs(env_kraken.z_interped))
ran =  np.linspace(0,
                   env_kraken.distances[-1]/1000,
                   num = BASIS_SIZE_distance) #my basis is in meters but Porter takes km.
depth = np.linspace(0,
                    1.5*bottom_max,
                    num = BASIS_SIZE_depth) #basis is in m, Porter takes m too for depth.
RX_DEPTH_INDEX = np.argmin(depth - RX_HYD_DEPTH < 0)
r = pyat.pyat.env.Dom(ran, depth)
pos = pyat.pyat.env.Pos(s, r)


modes, bandwidth, Pos1,pressure = \
    KRAKEN_functions.calculate_modes_and_pressure_field(
        env_kraken,
        pos)
TL_RES_KRAK = (pressure[0,0,RX_DEPTH_INDEX,:])

TL_read_back = TL_RES_BELL
R_read_back = HYD_1_R

# import pickle
# # with open('outputfile_TL', 'wb') as temp:
# #     pickle.dump(TL_RES_BELL, temp)
# # with open('outputfile_R', 'wb') as temp:
# #     pickle.dump(HYD_1_R, temp)

    
# with open ('outputfile_TL', 'rb') as temp:
#     TL_read_back = pickle.load(temp)
# with open ('outputfile_R', 'rb') as temp:
#     R_read_back = pickle.load(temp)
    
# plt.imshow(pressure[0,0,:,:]) 
    
y_bell = np.array(TL_read_back)
y_krak = np.array(TL_RES_KRAK)
missing = len(y_krak)-len(y_bell)
y_krak = y_krak[:len(y_krak)-missing]
y_krak = y_krak[::-1]
plt.plot(R_read_back,20*np.log10(np.abs(y_bell)),label='Bellhop')
plt.plot(R_read_back,y_krak,label='Kraken')
plt.plot(R_read_back,-20*np.log10(R_read_back),label='20logR')
plt.legend()