
import numpy as np
import matplotlib.pyplot as plt

import haversine
import arlpy.uwapm as pm

import sys
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


LOCATION = 'NS'
# LOCATION = 'BC'

#close enough
EMERALD_LAT = 44
EMERALD_LON = -62.5
MAX_LOCAL_DEPTH = 200 #A guess based on experience.

lat_extent_tuple = (EMERALD_LAT - 1, EMERALD_LAT + 1)
lon_extent_tuple = (EMERALD_LON - 1, EMERALD_LON + 1)

BASIS_SIZE = 5000
N_BEAMS = 100
FREQ = 500

RX_HYD_DEPTH = 50 #m

IR_RESULTS = []
TL_RES_BELL = []
TL_RES_KRAK = []
R_KRAK = []
HYD_1_R = []
HYD_2_R = []

KRAKEN_ROUGHNESS = [0.1,0.1]
IMPULSE_RESPONSE_REL_TX_TIME = True

if LOCATION == 'NS':
    # NS COORDINATES AND FILES
    hyd_1_lat   = 44.60551039
    hyd_1_lon   = -63.54031211
    hyd_2_lat   = 44.60516000
    hyd_2_lon   = -63.54272550    
    fname_bathy_NS = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
    fname_3rd_order_coef = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
    bottom_id  ='Sand-silt'
    
    
bathymetry = Bathymetry_WOD()
bathymetry.read_bathy(fname_bathy_NS)
bathymetry.sub_select_by_latlon(
    p_lat_extent_tuple = lat_extent_tuple,
    p_lon_extent_tuple = lon_extent_tuple) #has default values for NS already
bathymetry.interpolate_bathy()

#ssp = SSP_Blouin_2015()
ssp = SSP_Munk()
ssp.set_depths(np.linspace(0,MAX_LOCAL_DEPTH,BASIS_SIZE))
ssp.read_profile(fname_3rd_order_coef)
# ssp = SSP_Isovelocity()
#ssp.set_ssp(1485)

bottom_profile = SeaBed(bathymetry)
bottom_profile.read_default_dictionary()
bottom_profile.assign_single_bottom_type(bottom_id)

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
THE_SOURCE.set_depth()
THE_SOURCE.set_speed()
THE_SOURCE.generate_course(((EMERALD_LAT),(EMERALD_LON)),
                        p_CPA_deviation_m = 1500,
                        p_CPA_deviation_heading=haversine.Direction.EAST,
                        p_course_heading=1.99*np.pi, #(0,2pi), mathematical not navigation angles
                        p_distance=2000,
                        p_divisions = 10)

for TX_SOURCE in THE_SOURCE.course:
    # BELLHOP first, then KRAKEN afterwards
    env_bellhop = env_ARL.create_environment_model(
            (EMERALD_LAT,EMERALD_LON),
            TX_SOURCE,
            FREQ_TO_RUN = FREQ,
            RX_HYD_DEPTH = RX_HYD_DEPTH,
            TX_DEPTH = THE_SOURCE.depth,
            N_BEAMS = N_BEAMS,
            BASIS_SIZE = BASIS_SIZE
            )
    rays = pm.compute_eigenrays(env_bellhop)
    arrivals = pm.compute_arrivals(env_bellhop)
    ir = pm.arrivals_to_impulse_response(
        arrivals,
        fs=204800,
        abs_time = IMPULSE_RESPONSE_REL_TX_TIME
        )
    IR_RESULTS.append(ir)    
    TL = pm.compute_transmission_loss(
        env_bellhop,
        mode=pm.coherent,
        )
    x_cmplx = TL.iloc(0)[0].iloc(0)[0]
    TL_RES_BELL.append(x_cmplx)
    HYD_1_R.append(env_bellhop['rx_range'])

    # Now KRAKEN
    env_kraken = env_PYAT.create_environment_model(
            (EMERALD_LAT,EMERALD_LON),
            TX_SOURCE,
            THE_SOURCE,
            p_beam = [],
            BASIS_SIZE = BASIS_SIZE,
            freq = FREQ
            )
    
    s = pyat.pyat.env.Source([THE_SOURCE.depth])
    bottom_max = np.max(abs(env_kraken.z_interped))
    ran =  np.arange(0,env_kraken.distances[-1]/1000,0.01 ) #my basis is in meters but Porter takes km.
    depth = np.arange(0,1.5*bottom_max,0.1) #basis is in m, Porter takes m too for depth.
    r = pyat.pyat.env.Dom(ran, depth)
    pos = pyat.pyat.env.Pos(s, r)
    
    R_KRAK.append(ran[-1])
    modes, bandwidth, Pos1,pressure = \
        KRAKEN_functions.calculate_modes_and_pressure_field(
            env_kraken,
            pos)
    TL_RES_KRAK.append(float(pressure[0,0,500,-1]))


y_bell = np.array(TL_RES_BELL)
y_krak = np.array(TL_RES_KRAK) # 10log10 applied to pressure, need another factor of 2 for intensity
plt.plot(HYD_1_R,20*np.log10(np.abs(y_bell)),label='Bellhop')
plt.plot(HYD_1_R,y_krak,label='Kraken')
plt.plot(HYD_1_R,-20*np.log10(HYD_1_R),label='20logR')
plt.legend()