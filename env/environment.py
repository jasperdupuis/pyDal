# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 12:11:39 2021

@author: Jasper
"""

import sys
import ast
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

import geopy.distance as Distance

import bathymetry
import seabed
import surface

# ARL extra modules
import arlpy.uwapm as pm
import haversine

# PYAT extra modules
sys.path.insert(1,r'C:\Users\Jasper\Desktop\MASC\python-packages\pyat')
import pyat.pyat.env

# PyRAM includes
from pyram.PyRAM import PyRAM

def create_basis_common(bathy, #custom class in this module
                     rx_lat_lon_tuple,
                     tx_lat_lon_tuple,
                     BASIS_SIZE_depth,
                     BASIS_SIZE_distance):
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
    
    
        lat_basis = np.linspace(south_most,north_most,num=BASIS_SIZE_distance)
        lon_basis = np.linspace(west_most,east_most,num=BASIS_SIZE_distance)
        z_interped = bathy.calculate_interp_bathy(
            lat_basis,
            lon_basis,
            p_grid=False)
        
        #Calculate the distance this environment spans, this is where meters is set
        basis_min = (min(lat_basis),min(lon_basis))
        basis_max = (max(lat_basis),max(lon_basis))
        total_distance = Distance.distance(
            basis_min,
            basis_max ).m
        distances = np.arange(len(z_interped))
        distances = distances * total_distance/(len(z_interped)-2)
            
        depths = np.abs(np.linspace(0,np.max(z_interped),BASIS_SIZE_depth))
        
        #distances in m, depths in m
        return total_distance,distances, z_interped, depths    


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
        
    def __init__(self,
                 p_location,
                 p_NUM_LAT = 100,
                 p_NUM_LON = 100,
                 p_NUM_SSP = 100,
                 ):
        self.bathymetry = r'not set'
        self.ssp = r'not set'
        self.surface = r'not set'
        self.bottom_profile = r'not set'
        self.LAT_N_PTS = p_NUM_LAT
        self.LON_N_PTS = p_NUM_LON
        self.SSP_N_PTS = p_NUM_SSP
        self.set_location_common(p_location)
        self.set_surface_common()
        self.set_hyd_height() #hard coded
        
    def set_model_save_directory(self,p_dir):
        self.model_target_dir = p_dir
        
    def set_hyd_height (self,p_height = 1):
        self.hyd_height = p_height
        
    def set_bathymetry_common(self,p_bathymetry):
        self.bathymetry = p_bathymetry
        
        
    def set_ssp_common(self,p_ssp):
        """
        self.bathymetry must already be assigned before using this.
        """
        z = self.bathymetry.calculate_interp_bathy(
            self.bathymetry.lat_basis_trimmed,
            self.bathymetry.lon_basis_trimmed)
        MAX_DEPTH = np.abs (np.min( z ) - 1)  # negative is below sea level.
        p_ssp.set_depths(np.linspace(0, MAX_DEPTH+2, self.SSP_N_PTS))
        p_ssp.read_profile(self.location.ssp_file)
        self.ssp = p_ssp
        self.set_ssp()
        
        
    def set_seabed_common(
            self            
            ):
        bottom_profile = seabed.SeaBed(self.bathymetry.lat_basis_trimmed,
                                self.bathymetry.lon_basis_trimmed,
                                self.bathymetry.z_interped)
        bottom_profile.read_default_dictionary()
        bottom_profile.assign_single_bottom_type(self.location.bottom_id)
        self.bottom_profile = bottom_profile
        self.set_seabed()

    
    def set_surface_common(self):
        self.surface = surface.Surface()

    def set_rx_location_common(self,
        p_latlon_tuple,
        ):
        self.rx_latlon = p_latlon_tuple
            
   
    def set_source_common(self,p_source):
        self.source = p_source
   
    def set_freqs_common (self,p_freqs):
        self.freqs = p_freqs
    

    # Functions that derived classes must implement, even if only to call
    # the _common methods, above. 
    def create_environment_model(self,
                                 rx_lat_lon_tuple,
                                 tx_lat_lon_tuple,
                                        **kwargs
                                        ):
        pass

    def set_ssp(self):
        pass
        
    def set_seabed(self):
        pass

    
    def set_bathymetry(self,bathy):
        pass
    
    def set_surface(self,surface):
        pass
    
    def set_location_common(self,p_location):
        self.location = p_location
    
    def set_rx_depth_common(self,p_depth):
        self.rx_depth = p_depth
        
    def calculate_exact_TLs_common(self,**kwargs):
        pass

    
class Environment_RAM(Environment):
    
    def set_calc_params(self,
                        dr):
        self.DELTA_R_RAM = dr
    
    def set_ssp(self,
                roughness = [0.5,0.5],
                ssp_selection = 'Summer',
                ssp_ranges = np.array([1]) # in m
                ):   
        """
        """
        self.ssp_depths = self.ssp.depths
        self.ssp_r = ssp_ranges
        self.ssp_c = self.ssp.dict[ssp_selection]
        self.ssp_c = np.reshape(self.ssp_c,
                                (len(self.ssp_c),len(self.ssp_r))
                                )
       
        
    def set_seabed(self):
        """
        Takes the arrays in seabed_drdc.bottom_type_profile and takes the first value
        In the future may want to account for non-uniformity.
        But, parameter studies by others 
        indicate that isn't an issue compared to geometry and SSP
        
        """      
        self.seabed_rho = self.bottom_profile.bottom_type_profile['Rho'][0][0]
        self.seabed_c = self.bottom_profile.bottom_type_profile['c'][0][0]        
        self.seabed_alpha = self.bottom_profile.bottom_type_profile['alpha'][0][0]
    

    def create_environment_model(self,
                                 rx_lat_lon_tuple,
                                 tx_lat_lon_tuple,
                                 **kwargs):
        """
        Wrap the RAM code for generic environment.
        Because it is poorly separated need to feed the other 
        main inputs to this: ssp, source, bottom and surface objects.
        
        #Takes np.abs of the depth and distances, as model takes down to be positive.
        """
        self.freq = kwargs['FREQ_TO_RUN']
        self.source_depth = kwargs['TX_DEPTH']    # in m
        self.receiver_depth = kwargs['RX_DEPTH'] # in m
        
        #establishes the 2d slice bathymetry with range.
        total_distance,self.distances,self.z_interped,self.depths = \
            create_basis_common(
                self.bathymetry,
                rx_lat_lon_tuple,
                tx_lat_lon_tuple,
                kwargs['BASIS_SIZE_DEPTH'],
                kwargs['BASIS_SIZE_DISTANCE'])
            
        self.z_interped = np.abs(self.z_interped)
            
        # zips two equal length 1D vectors
        # add the .T argument at end to take transpose
        # which makes for two columns instead of two rows.
        self.bathy_2d_profile = np.vstack(
            (self.distances,self.z_interped)).T
        
        self.sb_update_z = np.zeros(1)
        self.sb_update_r = self.distances
            
        self.sb_rho_arr = self.seabed_rho * np.ones(len(self.distances))
        self.sb_rho_arr = np.reshape(
            self.sb_rho_arr,
            (len(self.sb_update_z),len(self.sb_update_r)))
        
        self.sb_c_arr = self.seabed_c * np.ones(len(self.distances))
        self.sb_c_arr = np.reshape(
            self.sb_c_arr,
            (len(self.sb_update_z),len(self.sb_update_r)))
        
        self.sb_alpha_arr = self.seabed_alpha * np.ones(len(self.distances))
        self.sb_alpha_arr = np.reshape(
            self.sb_alpha_arr,
            (len(self.sb_update_z),len(self.sb_update_r)))


        self.pyram_obj = PyRAM(
            self.freq,
            self.source_depth,
            self.receiver_depth,
            self.ssp_depths,
            self.ssp_r,
            self.ssp_c,
            self.sb_update_z,
            self.sb_update_r,
            self.sb_c_arr,
            self.sb_rho_arr,
            self.sb_alpha_arr,
            self.bathy_2d_profile,
            dr = self.DELTA_R_RAM
            )
        
        return
    
    def run_model(self):
        """
        From PyRAM module:
        results = {'ID': self._id,
              'Proc Time': self.proc_time,
              'Ranges': self.vr,
              'Depths': self.vz,
              'TL Grid': self.tlg,
              'TL Line': self.tll,
              'CP Grid': self.cpg,
              'CP Line': self.cpl,
              'c0': self._c0}
        """
        result = self.pyram_obj.run()
        # result is a dictionary. 
        # result['TL Line'] is the transmission loss profile at receiver depth.
        return result

    def calculate_exact_TLs(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        flat_earth_approx = Approximations() # didnt define static method

        for freq in self.freqs:
            TL_RES = []
            LAT = []
            LON = []

            for TX_SOURCE in self.source.course:
                # South hydrophone
                self.create_environment_model(
                            self.rx_latlon,
                            TX_SOURCE,
                            TX_DEPTH = self.source.depth,
                            FREQ_TO_RUN = freq,
                            RX_DEPTH = self.rx_depth,
                            BASIS_SIZE_DEPTH = kwargs['BASIS_SIZE_DEPTH'],
                            BASIS_SIZE_DISTANCE = kwargs['BASIS_SIZE_DISTANCE'],
                        )
                results_RAM = self.run_model()
                
                results_RAM['X'],results_RAM['Y'] = \
                    flat_earth_approx.RAM_distances_to_latlon(
                        p_cpa = (self.location.LAT,self.location.LON), 
                        p_rx = self.rx_latlon, 
                        p_tx = TX_SOURCE, 
                        p_r = results_RAM['Ranges'])
                
                results_RAM['TX Lat'] = TX_SOURCE[0]
                results_RAM['TX Lon'] = TX_SOURCE[1]
                
                TL_RES.append(results_RAM)
                LAT.append(TX_SOURCE[0])
                LON.append(TX_SOURCE[1])
                                
            self.RAM_dictionaries_to_unstruc(TL_RES)
                
            df_res = pd.DataFrame(
                data= {'X' : self.X_unstruc,
                       'Y' : self.Y_unstruc,
                       'TL': self.TL_unstruc,
                       'Lats TX' : self.lats_TX_unstruc,
                       'Lons TX' : self.lons_TX_unstruc,
                       })
            df_res.to_csv(
                self.model_target_dir       \
                    + str(freq).zfill(4)    \
                    + '.csv')
            

    def RAM_dictionaries_to_unstruc(self,p_dictionary_list):
        """
        The provided dictionaries are the default RAM outputs
        with the added X, Y, TX Lat, and TX Lon

        Upon generation the unstructured data is assigned to 
        object instance members.

        Parameters
        ----------
        p_dictionary_list : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        (Unstruct data sets X, Y, TL are assigned to 
        self.X_unstruc, self.Y_unstruc, self.TL_unstruc)

        """
        
        X_unstruc = []
        Y_unstruc = []
        TL_unstruc = []
        TX_lat = []
        TX_lon = []
        for dictionary in p_dictionary_list:
            X_unstruc.append(dictionary['X'])
            Y_unstruc.append(dictionary['Y'])
            TL_unstruc.append(dictionary['TL Line'])
            ones = np.ones_like(dictionary['X'])
            TX_lat.append(ones * dictionary['TX Lat'])
            TX_lon.append(ones * dictionary['TX Lon'])
        
        x = [item for sublist in X_unstruc for item in sublist]
        y = [item for sublist in Y_unstruc for item in sublist]
        TL = [item for sublist in TL_unstruc for item in sublist]
        lats = [item for sublist in TX_lat for item in sublist]
        lons = [item for sublist in TX_lon for item in sublist]
        
        
        self.X_unstruc = x
        self.Y_unstruc = y
        self.TL_unstruc = TL        
        self.lats_TX_unstruc = lats
        self.lons_TX_unstruc = lons
        

    def interpolate_RAM_data_obj(self,p_xlim,p_ylim):
        """
        Creates a TL profile over a target grid using the existing unstructured
        data.
        
        p_xlim and p_ylim specify the target grid for interpolation.

        This method uses class members as data sources.
        
        There is another method for file data sources.

        Returns
        -------
        None.

        """
        self.xlim_interp = p_xlim
        self.ylim_interp = p_ylim
        
        self.TL_interp = Environment_RAM.interpolate_RAM_data(
            self.X_unstruc,
            self.Y_unstruc,
            self.TL_unstruc,
            p_xlim,
            p_ylim)


    @staticmethod
    def interpolate_RAM_data(x,y,TL,xlim,ylim,n_step=100):
        x_basis = np.linspace(-1*xlim,xlim,n_step)
        y_basis = np.linspace(-1*ylim,ylim,n_step)
        x_target,y_target = np.meshgrid(x_basis,y_basis)
    
        source_points = ( x , y )
        xi = ( x_target , y_target )
        TL_interp = interpolate.griddata(
            source_points,
            TL,
            xi
            )
        return TL_interp,xi
        
    def plot_TL_interpolation_with_couse(self,
                              p_r,
                              p_x_track,
                              p_y_track):        
        x_comex = []
        y_comex = []
        for theta in range(360):
            theta = theta * np.pi / 180
            x_comex.append( np.cos(theta) * p_r )
            y_comex.append(np.sin(theta) * p_r )

        ext = [-1*self.xlim_interp,
               self.xlim_interp,
               -1*self.ylim_interp,
               self.ylim_interp ]
        plt.figure();plt.imshow(
            self.TL_interp,
            extent = ext,
            origin='lower',
            aspect='auto');
        plt.scatter(p_x_track,p_y_track,color='r',marker='.',label = 'Course')
        plt.scatter(x_comex,y_comex,color='c',marker='x',label='COMEX/FINEX circle')
        plt.legend()

    
class Environment_PYAT(Environment):
    
    def set_ssp(self,
                roughness = [0.5,0.5],
                ssp_selection = 'Summer'):   
        """
        """
        depths = [0,self.ssp.depths[-1]]
        ssp1 = pyat.pyat.env.SSPraw(
            self.ssp.depths, 
            self.ssp.dict[ssp_selection],
            0*np.ones(self.ssp.depths.shape), # water has no shear speed, betaR
            np.ones(self.ssp.depths.shape), # density of water = 1kg/L, rho
            0*np.ones(self.ssp.depths.shape), # water has no attenuation, alphaI
            0*np.ones(self.ssp.depths.shape)) # water has no shear attenuation, betaI

        raw = [ssp1]
        NMedia		=	1
        Opt			=	'CVW'	
        N			=	[self.ssp.depths.size]
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
        total_distance,self.distances,self.z_interped,self.depths = \
            create_basis_common(
                self.bathymetry,
                rx_lat_lon_tuple,
                tx_lat_lon_tuple,
                kwargs['BASIS_SIZE_depth'],
                kwargs['BASIS_SIZE_distance'])
        self.Z = np.abs( self.depths )# in m
        # self.Z = np.abs( self.z_interped )# in m
        self.X = np.abs( self.distances / 1000 )# in m converted to km
        self.s = pyat.pyat.env.Source(source_drdc.depth)
        self.r = pyat.pyat.env.Dom(self.X, self.Z) #needs to be in km, m
        self.pos = pyat.pyat.env.Pos(self.s,self.r)
        self.pos.s.depth	= [source_drdc.depth]
        self.pos.r.depth	 = self.Z
        self.pos.r.range		=	self.X
        
        return self
    
    
class Environment_ARL(Environment):

        
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
        # 20211118: Does the above line mean anything now??
        
        arguments:
            lat_basis
            lon_basis
            FREQ_TO_RUN
            RX_HYD_DEPTH
            RX_HYD_RANGE
            TX_DEPTH
        
        """
        env = pm.create_env2d()        
        
        total_distance,self.distances,self.z_interped,self.depths = \
            create_basis_common(
                self.bathymetry,
                rx_lat_lon_tuple,
                tx_lat_lon_tuple,
                kwargs['BASIS_SIZE_DEPTH'],
                kwargs['BASIS_SIZE_DISTANCE'])
             
        self.surface.SS_0(total_distance) #sea state 0, no params other than wave height.
        
        depth_array = []
        for r,z in zip(self.distances,self.z_interped):
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
    
    def calculate_exact_TLs(self,
                            **kwargs):
        count = 0
        TL_RES = []
        LAT = []
        LON = []
        for freq in self.freqs:
            for TX_SOURCE in self.source.course:
                env_bellhop_S = self.create_environment_model(
                        self.rx_latlon,
                        TX_SOURCE,
                        FREQ_TO_RUN = freq,
                        RX_HYD_DEPTH = self.rx_depth, 
                        TX_DEPTH = self.source.depth, 
                        N_BEAMS = kwargs['N_BEAMS'], 
                        BASIS_SIZE_DEPTH = kwargs['BASIS_SIZE_DEPTH'],
                        BASIS_SIZE_DISTANCE = kwargs['BASIS_SIZE_DISTANCE'],
                    )
                TL = pm.compute_transmission_loss(
                    env_bellhop_S,
                    mode=pm.coherent,
                    )
                
                try:
                    x_cmplx = TL.iloc(0)[0].iloc(0)[0]
                    # TL_RES_BELL_S[index] = np.abs(x_cmplx)
                    TL_RES.append(np.abs(x_cmplx))
                    LAT.append(TX_SOURCE[0])
                    LON.append(TX_SOURCE[1])
                except:
                    print('BELLHOP: error at range: ' +str(env_bellhop_S['rx_range']))
                    print ('BELLHOP: This is actually lat-lon: ' + str(TX_SOURCE[0]) + ' , ' + str (TX_SOURCE[1]))
                
                if count % 100 == 0: 
                    print('Still working on BELLHOP! Currently at:')
                    print('Freq: \t' + str ( freq ) + '\t Count: \t:' + str ( count ) )
                count = count + 1

        self.LAT = LAT
        self.LON = LON
        self.TL = 20 * np.log10(TL_RES)

    def reshape_1d_to_surface(self):
        
        X,Y = \
            self.bathymetry.convert_latlon_to_xy_m(
                self.location, 
                np.array(self.LAT),
                np.array(self.LON)
                )
                
        tl_surf = np.reshape(self.TL,
                             (self.LAT_N_PTS,self.LON_N_PTS)
                             )
        return X,Y,tl_surf
        

class Approximations():
    
    def __init__(self):
        self.R = Distance.EARTH_RADIUS * 1000 #km to m
        self.DEG_TO_RAD = np.pi / 180

    def latlon_to_xy(self,p_cpa,p_latlon):
        """
        Convert a single latlon point to x,y, 
        referred to the passed CPA as 0,0.
        
        Returns a tuple.
        
        Error about 0.25%

        Parameters
        ----------
        p_cpa : TYPE
            DESCRIPTION.
        p_latlon : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        lat,lon = p_latlon[0],p_latlon[1]
        lat_CPA, lon_CPA = p_cpa[0], p_cpa[1]
        
        x = self.R \
            * self.DEG_TO_RAD * (lon - lon_CPA)\
            * np.cos(np.mean(lat * self.DEG_TO_RAD ) )
        y = self.R * self.DEG_TO_RAD * (lat - lat_CPA) 

        return (x,y)


    def xy_to_latlon(self,p_cpa,p_xy):
        """
        Using p_cpa as the 0,0 xy reference, convert the passed xy
        point to lat-lon using flat earth approximation.

        Error about 0.25%

        Parameters
        ----------
        p_cpa : TYPE
            a reference lat,lon tuple that represents 0,0 in the xy system.
        p_xy : TYPE
            DESCRIPTION.

        Returns
        -------
        (lat,lon) tuple

        """
        
        x,y = p_xy[0],p_xy[1]
        lat,lon = p_cpa[0],p_cpa[1]
        lat = ( y / self.R ) + lat 
        lon = ( x / self.R * np.cos(lat) ) + lon
        return (lat,lon)


    def RAM_distances_to_latlon(self,
                                p_cpa,#latlon
                                p_rx, #latlon
                                p_tx, #latlon
                                p_r #in m
                                ):
        """
        With p_cpa as 0,0, take the p_rx and p_tx and find corresponding latlon
        points for each element of the passed p_r.
        
        This is done by building a new xy set representing p_r in xy, and then
        casting those to latlon.
        
        Parameters
        ----------
        p_cpa : TYPE
            DESCRIPTION.
        p_rx : TYPE
            DESCRIPTION.
        p_tx : TYPE
            DESCRIPTION.
        p_r : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        try:
            p_r = np.array(p_r) # ensure it's an array passed.
        except:
            print('Approximations.RAM_distances_to_latlon:\n \
                  Range was not passed in array-castable form')
        
        xy_rx = self.latlon_to_xy(p_cpa,p_rx)
        xy_tx = self.latlon_to_xy(p_cpa,p_tx)
        
        dx = xy_rx[0] - xy_tx[0]
        dy = xy_rx[1] - xy_tx[1]
        # Model as phi the angle from TX point to RX point relative to +ve x.
        # (Adjacent staying the same, only opposite should be  changing)
        phi = np.arctan2(dy,dx)
        x = np.cos(phi) * p_r
        xc = x + xy_tx[0]    

        y = np.sin(phi) * p_r
        yc = y + xy_tx[1]
        
        return xc,yc
