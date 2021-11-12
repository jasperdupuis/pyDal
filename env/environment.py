# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:11:39 2021

@author: Jasper
"""

import sys
import ast
import numpy as np
import matplotlib.pyplot as plt

# ARL extra modules
import arlpy.uwapm as pm
from geopy import distance
import haversine

# PYAT extra modules
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')
import pyat.pyat.env


class Empty():
    def __init__(self):
        return

class Environment():
    """
    The base class.
    Different models will likely have different input requirements,
    but underlying environment will not change. Derived classes can implement
    methods for specific requirements.
    """
    
    def __init__(self):
        self.bathymetry = r'not set'
        self.ssp = r'not set'
        self.surface = r'not set'
        self.bottom_profile = r'not set'
        
    def set_bathymetry_common(self,p_bathymetry):
        self.bathymetry= p_bathymetry
        
    def set_ssp_common(self,p_ssp):
        self.ssp = p_ssp
        
    def set_seabed_common(self,p_seabed):
        self.bottom_profile = p_seabed
    
    def set_surface_common(self,p_surface):
        self.surface = p_surface
 
    def create_basis_common(self,
                     rx_lat_lon_tuple,
                     tx_lat_lon_tuple,
                     basis_size):
        """
        Create a basis 2D environment, depths and distances.
        self.distances, self.z_interp in meters.
        """
        
        # the basis needs to be arranged in increasing order, so some logic.
        if rx_lat_lon_tuple[0] > tx_lat_lon_tuple[0]:
            north_most = rx_lat_lon_tuple[0]
            south_most = tx_lat_lon_tuple[0]
        else:
            north_most = tx_lat_lon_tuple[0]
            south_most = rx_lat_lon_tuple[0]

        #Be careful of sign for lon: -180,180.
        if rx_lat_lon_tuple[1] > tx_lat_lon_tuple[1]:
            east_most = rx_lat_lon_tuple[1]
            west_most = tx_lat_lon_tuple[1]
        else:
            east_most = tx_lat_lon_tuple[1]
            west_most = rx_lat_lon_tuple[1]

        lat_basis = np.linspace(south_most,north_most,num=basis_size)
        lon_basis = np.linspace(west_most,east_most,num=basis_size)
        z_interped = self.bathymetry.calculate_interp_bathy(lat_basis,lon_basis)
        
        #Calculate the distance this environment spans, thi sis where meters is set
        basis_min = (min(lat_basis),min(lon_basis))
        basis_max = (max(lat_basis),max(lon_basis))
        total_distance = distance.distance(
            basis_min,
            basis_max ).m
        distances = np.arange(len(z_interped))
        distances = distances * total_distance/(len(z_interped)-2)
    
        self.distances = distances #in m
        self.z_interped = z_interped
        
        depths = np.linspace(0,np.max(z_interped),basis_size)
        self.depth_linspace = depths
        
        return total_distance,distances, z_interped, depths
    
    # Functions that derived classes must implement, even if only to call
    # the _common methods, above. 
    def create_environment_model(self,
                                 rx_lat_lon_tuple,
                                 tx_lat_lon_tuple,
                                        **kwargs
                                        ):
        pass

    def set_ssp(self,ssp):
        pass
        
    def set_seabed(self, p_seabed):
        pass
    
    def set_bathymetry(self,bathy):
        pass
    
    def set_surface(self,surface):
        pass
       
    
class Environment_PYAT(Environment):
    
    def set_ssp(self,
                ssp_drdc,
                roughness = [0.5,0.5],
                ssp_selection = 'Summer'):   
        """
        """
        self.set_ssp_common(ssp_drdc)
        depths = [0,ssp_drdc.depths[-1]]
        ssp1 = pyat.pyat.env.SSPraw(
            ssp_drdc.depths, 
            ssp_drdc.dict[ssp_selection],
            0*np.ones(ssp_drdc.depths.shape), # water has no shear speed, betaR
            np.ones(ssp_drdc.depths.shape), # density of water = 1kg/L, rho
            0*np.ones(ssp_drdc.depths.shape), # water has no attenuation, alphaI
            0*np.ones(ssp_drdc.depths.shape)) # water has no shear attenuation, betaI

        raw = [ssp1]
        NMedia		=	1
        Opt			=	'CVW'	
        N			=	[ssp_drdc.depths.size]
        sigma		=	[.5,.5]	 # roughness at each layer. only effects attenuation (imag part)
        self.ssp_pyat = pyat.pyat.env.SSP(raw, depths, NMedia, Opt, N, sigma)
    
    def set_seabed(self,
                   seabed_drdc,
                   p_betaR = 0,
                   p_betaI = 0):
        """
        Takes the arrays in seabed_drdc.bottom_type_profile and takes the first value
        In the future may want to account for non-uniformity.
        But, parameter studies indicate that isn't an issue compared to geometry and SSP
        
        ALSO SETS SURFACE BOUNDARY CONDITION
        #TODO: Separate this from the bottom setting.
        """
        self.set_seabed_common(seabed_drdc)
        hs = pyat.pyat.env.HS(
            alphaR=seabed_drdc.bottom_type_profile['c'][0][0],
            betaR=p_betaR, #unknown what this is
            rho = seabed_drdc.bottom_type_profile['Rho'][0][0],
            alphaI=seabed_drdc.bottom_type_profile['alpha'][0][0],
            betaI=p_betaI) #unknow what this is
        Opt = 'A~'
        bottom = pyat.pyat.env.BotBndry(Opt, hs)
        top = pyat.pyat.env.TopBndry('CVW')
        self.bdy = pyat.pyat.env.Bndry(top, bottom)

        self.cInt = Empty()
        self.cInt.High = seabed_drdc.bottom_type_profile['c'][0][0]+1 #assumes 2d array passed.
        self.cInt.Low = 0 # compute automatically, not sure what this means.
    
    def set_bathymetry(self,bathy):
        self.set_bathymetry_common(bathy)
    
    def set_surface(self,surface):
        self.set_surface_common(surface)
    
    def create_environment_model(self,
                                 rx_lat_lon_tuple,
                                 tx_lat_lon_tuple,
                                 source_drdc,
                                 p_beam = [], #dont know this default value meaning
                                 **kwargs):
        """
        Wrap the PYAT test case for generic environment.
        Because it is poorly separated need to feed the other 
        main inputs to this: ssp, source, bottom and surface objects.
        
        #Takes np.abs of the depth and distances, as model takes down to be positive.
        """
        self.freq = kwargs['freq']
        self.beam = p_beam
        total_distance,distances,z_interp,depths = self.create_basis_common(
                    rx_lat_lon_tuple,
                    tx_lat_lon_tuple,
                    kwargs['BASIS_SIZE'])
        self.Z = np.abs( depths )# in m
        self.X = np.abs( distances / 1000 )# in m converted to km
        self.s = pyat.pyat.env.Source(source_drdc.depth)
        self.r = pyat.pyat.env.Dom(self.X, self.Z) #needs to be in km, m
        self.pos = pyat.pyat.env.Pos(self.s,self.r)
        self.pos.s.depth	= [source_drdc.depth]
        self.pos.r.depth	 = self.Z
        self.pos.r.range		=	self.X
        
        return self
    
    
class Environment_ARL(Environment):

    def set_ssp(self,ssp):
        self.set_ssp_common(ssp)
        
    def set_seabed(self, p_seabed):
        self.set_seabed_common(p_seabed)
        
    def set_bathymetry(self,bathy):
        self.set_bathymetry_common(bathy)
        
    def set_surface(self,surface):
        self.set_surface_common(surface)

    def create_environment_model(self,
                                 rx_lat_lon_tuple,
                                 tx_lat_lon_tuple,
                                        **kwargs
                                        ):
        """
        Creates an arlpy Acoustic Research Lab environment object.
        Note this depends on the linspace in main function having 50 entries.
        
        arguments:
            lat_basis
            lon_basis
            FREQ_TO_RUN
            RX_HYD_DEPTH
            RX_HYD_RANGE
            TX_DEPTH
        
        """
        env = pm.create_env2d()        
        
        total_distance,distances,z_interped,depths = self.create_basis_common(
                        rx_lat_lon_tuple,
                        tx_lat_lon_tuple,
                        kwargs['BASIS_SIZE'])
             
        self.surface.SS_0(total_distance) #sea state 0, no params other than wave height.
        self.set_surface(self.surface)
        
        depth_array = []
        for r,z in zip(distances,z_interped):
            depth_array.append([r,np.abs(z)])
        depth_array = np.array(depth_array)
        
        ssp_array = []
        for z,ssp in zip (self.ssp.depths,self.ssp.get_summer()):
            ssp_array.append([np.abs(z),ssp])
        ssp_array = np.array(ssp_array)
        
        
        # CREATE AND POPULATE THE BELLHOP ENVIRONMENT
        # NOTE THAT THE 2D PROFILE ARRAYS ARE REDUCED TO A SCALAR BY [0,0]
        
        #env['name']
        env['bottom_absorption'] = self.bottom_profile.bottom_type_profile['alpha'][0,0]
        env['bottom_density'] = self.bottom_profile.bottom_type_profile['Rho'][0,0]
        #env['bottom_roughness']
        env['bottom_soundspeed'] = self.bottom_profile.bottom_type_profile['c'][0,0]
        env['depth'] = depth_array
        # env['depth_interp']
        env['frequency'] = kwargs['FREQ_TO_RUN']
        # env['max_angle']
        # env['min_angle']
        env['nbeams'] = kwargs['N_BEAMS']
        env['rx_depth'] = kwargs['RX_HYD_DEPTH']
        env['rx_range'] = total_distance                                  
        env['soundspeed'] = ssp_array
        # env['soundspeed_interp']
        env['surface'] = self.surface.surface_desc
        # env['surface_interp']
        env['tx_depth'] = kwargs['TX_DEPTH']
        # env['tx_directionality'] 
        # env['type'] 
    
        self.env = env
    
        return env
    