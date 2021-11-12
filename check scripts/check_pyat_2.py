# Anaconda distro modules
import numpy as np
from os import system
from matplotlib import pyplot as plt

#downloaded modules
import haversine
import sys #have to put on path even if unused here.
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')
import pyat

#my modules
from env.environment import Environment_ARL,Environment_PYAT
from env.bathymetry import Bathymetry_WOD
from env.ssp import SSP_Blouin_2015, SSP_Munk, SSP_Isovelocity
from env.seabed import SeaBed
from env.surface import Surface


import source
import KRAKEN_functions

BASIS_SIZE = 1000
N_BEAMS = 100
FREQ = 1000

RX_HYD_DEPTH = 20.9 #m

IR_RESULTS = []
TL_RESULTS = []
HYD_1_R = []
HYD_2_R = []

KRAKEN_ROUGHNESS = [0.1,0.1]
IMPULSE_RESPONSE_REL_TX_TIME = True

# NS COORDINATES AND FILES
hyd_1_lat   = 44.60551039
hyd_1_lon   = -63.54031211
hyd_2_lat   = 44.60516000
hyd_2_lon   = -63.54272550    
fname_bathy_NS = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
fname_3rd_order_coef = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
bottom_id  ='Sand-silt'


CPA_lat = ( hyd_1_lat + hyd_2_lat ) / 2
CPA_lon = ( hyd_1_lon + hyd_2_lon ) / 2


#for reducing the large WOD data to a small window on range
if hyd_1_lat > hyd_2_lat: #hyd 1 is more north than hyd 2
    lat_extent_tuple = (hyd_2_lat-0.01,hyd_1_lat+0.01)
else: lat_extent_tuple = (hyd_1_lat-0.01,hyd_2_lat+0.01)
if hyd_1_lon > hyd_2_lon: #hyd 1 is more east than hyd 2
    lon_extent_tuple =  (hyd_2_lon-0.01,hyd_1_lon+0.01)
else: lon_extent_tuple =  (hyd_1_lon-0.01,hyd_2_lon+0.01)

# # test a fixed depth for debuggin only.
# bathymetry = environment.Bathymetry_range_independent_pekeris()
# bathymetry.set_depth(fname_bathy_NS,depth=-25) #25m
# bathymetry.set_lat_lon_basis(10e3) #10km
# Fully varying profile
bathymetry = Bathymetry_WOD()
bathymetry.read_bathy(fname_bathy_NS)
bathymetry.sub_select_by_latlon(
    p_lat_extent_tuple = lat_extent_tuple,
    p_lon_extent_tuple = lon_extent_tuple) #has default values for NS already
bathymetry.interpolate_bathy()


#ssp = SSP_Blouin_2015()
ssp = SSP_Munk()
ssp.set_depths(np.linspace(0,np.abs(np.min(bathymetry.z_selection)-1),BASIS_SIZE))
ssp.read_profile(fname_3rd_order_coef)
# ssp = SSP_Isovelocity()
#ssp.set_ssp(1485)

bottom_profile = SeaBed(bathymetry)
bottom_profile.read_default_dictionary()
bottom_profile.assign_single_bottom_type(bottom_id)

surface = Surface()

THE_ENVIRONMENT = Environment_PYAT()
THE_ENVIRONMENT.set_bathymetry(bathymetry)
THE_ENVIRONMENT.set_seabed(bottom_profile)
THE_ENVIRONMENT.set_ssp(ssp,KRAKEN_ROUGHNESS)

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

TX_SOURCE = THE_SOURCE.course[0]
env = THE_ENVIRONMENT.create_environment_model(
            (hyd_1_lat,hyd_1_lon),
            TX_SOURCE,
            THE_SOURCE,
            p_beam = [],
            BASIS_SIZE = BASIS_SIZE,
            freq = FREQ
            )


s = pyat.pyat.env.Source([THE_SOURCE.depth])
bottom_max = np.max(abs(env.z_interped))
ran =  np.arange(0,1, 10/1e3)
depth = np.arange(0,1.5*bottom_max,0.1)
r = pyat.pyat.env.Dom(ran, depth)
pos = pyat.pyat.env.Pos(s, r)

modes, bandwidth, Pos1,pressure = \
    KRAKEN_functions.calculate_modes_and_pressure_field(
        env,
        pos)


KRAKEN_functions.plot_and_show_result(Pos1, pressure)
