import numpy as np
import matplotlib.pyplot as plt

import haversine
import arlpy.uwapm as pm

import sys
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')
import pyat

#my modules
from .environment import create_basis_common,Environment_ARL,Environment_PYAT,Environment_RAM
from .bathymetry import Bathymetry_WOD
from .ssp import SSP_Blouin_2015, SSP_Munk, SSP_Isovelocity
from .seabed import SeaBed
from .surface import Surface
from .locations import Location

import source
import KRAKEN_functions

def setup(the_location,
          p_source_depth = 4, #default for propeller center
          p_course_heading = 1.99,
          p_course_num_points = 50,
          p_pekeris_depths = 100,
          p_basis_size_depth = 5000,
          p_basis_size_range = 5000,
          p_kraken_roughness = [0.5,0.5],          
          p_ram_delta_r = 50, # calculation range step, default in pyram is freq * num_pade_terms
          p_depth_offset = 0, #in meters, positive
          p_CPA_offset = 0 # in meters, positive.
          ):
          
    
    bathymetry = Bathymetry_WOD()
    bathymetry.read_bathy(the_location.fname_bathy)
    bathymetry.sub_select_by_latlon(
        p_lat_extent_tuple = the_location.LAT_EXTENT_TUPLE,
        p_lon_extent_tuple = the_location.LON_EXTENT_TUPLE) #has default values for NS already
    # If pekeris, set the depth uniformly over the space.
    if the_location.location_title == 'Pekeris Waveguide':
        #set the constant depth
        bathymetry.z = p_pekeris_depths*np.ones_like(bathymetry.z)
        bathymetry.z_selection= p_pekeris_depths * np.ones_like(bathymetry.z_selection)

    #APPLY  DEPTH OFFSET - correct for over achieving curve fit and overly granular data
    bathymetry.z_selection = bathymetry.z_selection - p_depth_offset # z negative ==> below sea level at this point.

    bathymetry.interpolate_bathy()
    
    
    # set source properties
    THE_SOURCE = source.Source()
    THE_SOURCE.set_name()
    THE_SOURCE.set_depth(p_source_depth) #Default is 4m
    THE_SOURCE.set_speed()
    THE_SOURCE.generate_course((the_location.LAT,the_location.LON),
                            p_CPA_deviation_m = p_CPA_offset,
                            p_CPA_deviation_heading=haversine.Direction.EAST,
                            p_course_heading=p_course_heading*np.pi, #(0,2pi), mathematical not navigation angles
                            p_distance= the_location.COURSE_DISTANCE, # m
                            p_divisions = p_course_num_points) #number of divisions
    

    total_distance,distances, z_interped, depths = \
        create_basis_common(
            bathymetry,
            THE_SOURCE.course[0],
            THE_SOURCE.course[-1],
            p_basis_size_depth,
            p_basis_size_range)
    
    # THIS IS KIND OF OUT OF ORDER, BUT NEED IT HERE FOR KOSHER SSP WITH KRAKEN
    MAX_LOCAL_DEPTH = np.abs(np.min(z_interped))
    MAX_LOCAL_DEPTH +=1 # THIS IS A HACK TO MAKE SURE SSP EXTENDS PAST BOTTOM.
    
    if the_location.location_title =='Pekeris Waveguide':
        ssp = SSP_Isovelocity()
        ssp.set_depths(np.linspace(0,MAX_LOCAL_DEPTH,p_basis_size_depth))
        ssp.read_profile(the_location.ssp_file)

    if the_location.location_title =='Ferguson Cove, NS':
        ssp = SSP_Blouin_2015()
        ssp.set_depths(np.linspace(0,MAX_LOCAL_DEPTH,p_basis_size_depth))
        ssp.read_profile(the_location.ssp_file)

    if the_location.location_title =='Emerald Basin, NS':
        ssp = SSP_Munk()
        ssp.set_depths(np.linspace(0,MAX_LOCAL_DEPTH,p_basis_size_depth))
        ssp.read_profile(the_location.ssp_file)
    
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
    env_PYAT.set_ssp(ssp,p_kraken_roughness)
    
    env_RAM = Environment_RAM()  
    env_RAM.set_bathymetry(bathymetry)
    env_RAM.set_seabed(bottom_profile)
    env_RAM.set_ssp(ssp)
    env_RAM.set_calc_params(p_ram_delta_r)

    return THE_SOURCE, env_ARL, env_PYAT, env_RAM
