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

import source

DEG_OFFSET = 0.75 # Each location has a centre point, this defines the decimal degrees to each side from that.
# LOCATION = 'Emerald Basin'
# LOCATION = 'Pat Bay'
# LOCATION = 'Ferguson Cove'
LOCATION = 'Pekeris Waveguide'
the_location = Location(LOCATION,DEG_OFFSET)
    

# Parameters to manipulate
FREQ = 500
RX_HYD_DEPTH = 50 #m
num_COURSE_POINTS = 2
BASIS_SIZE_depth = 5000
BASIS_SIZE_distance = 5000
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
                        p_divisions = num_COURSE_POINTS) #number of divisions

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

env_RAM = Environment_RAM()  
env_RAM.set_bathymetry(bathymetry)
env_RAM.set_seabed(bottom_profile)
env_RAM.set_ssp(ssp)

#Generate the Bellhop result which requires recomputing every move.
lat_end = THE_SOURCE.course[-1][0]
lon_end = THE_SOURCE.course[-1][1]
count = 0

for TX_SOURCE in THE_SOURCE.course:
    if TX_SOURCE[0] == lat_end: break
    # BELLHOP first, then KRAKEN afterwards
    env_RAM.create_environment_model(
            (lat_end,lon_end),
            TX_SOURCE,
            TX_DEPTH = THE_SOURCE.depth,
            FREQ_TO_RUN = FREQ,
            RX_DEPTH = RX_HYD_DEPTH,
            BASIS_SIZE_DEPTH = BASIS_SIZE_depth,
            BASIS_SIZE_DISTANCE = BASIS_SIZE_distance,
        )
    results_RAM = env_RAM.run_model()
    print(str(count) +': '+ str(TX_SOURCE))
    count+=1
    
results_RAM['Depths']

plt.plot(results_RAM['TL Line'][::-1])
