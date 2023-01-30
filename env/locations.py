# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 10:14:40 2021

@author: Jasper
"""

import numpy as np

class Location():
    
    def __init__(self,
                 env_string):
        self.load_location(env_string)
    
    def load_location(self,
                      environment_string,
                      ):
        """
        Manage lat/lon coordinates

        Recommended lat/lon dec degree offsets are hard coded:
            Emerald basis: 0.75 deg
            Ferguson Cove: 0.01 deg 
                - note this is for 10m resolution files, can go smaller if needed.
                - the 200m between hyds is about 0.0035 dec deg.
                
        Parameters
        ----------
        environment_string : TYPE
            DESCRIPTION.
        dec_deg : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if environment_string == 'Emerald Basin':
            self.location_title = 'Emerald Basin, NS'
            self.DEG_OFFSET = 0.75
            self.COURSE_DISTANCE = 5000
            self.LAT = 44
            self.LON = -62.5
            self.LAT_EXTENT_TUPLE = (self.LAT - self.DEG_OFFSET , self.LAT + self.DEG_OFFSET )
            self.LON_EXTENT_TUPLE = (self.LON - self.DEG_OFFSET , self.LON + self.DEG_OFFSET )
            self.fname_bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
            self.ssp_file = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
            self.bottom_id  ='Sand-silt'
            

        if environment_string == 'Ferguson Cove':
            self.location_title = 'Ferguson Cove, NS'
            self.DEG_OFFSET = 0.005 #about 0.5k at equator. 
            self.COURSE_DISTANCE = 300
            self.hyd_1_lat   = 44.60551039
            self.hyd_1_lon   = -63.54031211
            self.hyd_2_lat   = 44.60516000
            self.hyd_2_lon   = -63.54272550
            self.LAT = ( self.hyd_1_lat + self.hyd_2_lat ) / 2 # This is CPA
            self.LON = ( self.hyd_1_lon + self.hyd_2_lon ) / 2 # This is CPA
            self.LAT_EXTENT_TUPLE = (self.LAT - self.DEG_OFFSET , self.LAT + self.DEG_OFFSET )
            self.LON_EXTENT_TUPLE = (self.LON - self.DEG_OFFSET , self.LON + self.DEG_OFFSET )
            self.fname_bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_GEBCO/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
            self.ssp_file = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
            self.bottom_id  ='Sand-silt'
            
        if environment_string == 'Patricia Bay':
            self.location_title = 'Patricia Bay, BC'
            # self.EXTENT_DEG_OFFSET = 0.005
            self.EXTENT_DEG_OFFSET = 0.005
            self.CORRIDOR_LON_OFFSET = 0.003
            self.CORRIDOR_LAT_OFFSET = 0.0005
            self.COURSE_DISTANCE = 300
            self.hyd_1_lat   = 48.65916667 # north hyd
            self.hyd_1_lon   = -123.47860000
            self.hyd_1_z     = 24.1
            self.hyd_2_lat   = 48.65765000 #south hyd
            self.hyd_2_lon   = -123.48006667
            self.hyd_2_z     = 41.2
            self.LAT = ( self.hyd_1_lat + self.hyd_2_lat ) / 2
            self.LON = ( self.hyd_1_lon + self.hyd_2_lon ) / 2
            # 
            """
            EXTENT is a geographic parameter, ideally an N x N box which 
            includes the entirety of the runs, the hydrophones, and a bit
            more beyond the bounds thereby defined.
            """
            self.LAT_EXTENT_TUPLE = (
                np.minimum(self.hyd_1_lat,self.hyd_2_lat) - 0.0005,
                np.maximum(self.hyd_1_lat,self.hyd_2_lat) + 0.0005)
            self.LON_EXTENT_TUPLE = ( 
                self.LON - self.EXTENT_DEG_OFFSET,
                self.LON + self.EXTENT_DEG_OFFSET)
            """
            In contrast the RANGE_CORRIDOR is that much smaller area which 
            describes the area over which the ship might realistically pass.
            It is usually around 200x50m in size.
            """
            self.LAT_RANGE_CORRIDOR_TUPLE = (
                self.LAT - self.CORRIDOR_LAT_OFFSET,
                self.LAT + self.CORRIDOR_LAT_OFFSET)

            self.LON_RANGE_CORRIDOR_TUPLE = ( 
                self.LON - self.CORRIDOR_LON_OFFSET,
                self.LON + self.CORRIDOR_LON_OFFSET)
            

            self.fname_bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/BC_CHS_2m/2m_Combine_PatBay.txt'
            self.ssp_file = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
            self.bottom_id  ='Sand-silt'
            
            
        if environment_string == 'Pekeris Waveguide': #use the NS bathy for a basis nd set it to desired depth elsewhere.
            self.location_title = 'Pekeris Waveguide'
            self.COURSE_DISTANCE = 5000
            self.DEG_OFFSET = 1.5
            self.LAT = 44
            self.LON = -62.5
            self.LAT_EXTENT_TUPLE = (self.LAT - self.DEG_OFFSET , self.LAT + self.DEG_OFFSET )
            self.LON_EXTENT_TUPLE = (self.LON - self.DEG_OFFSET , self.LON + self.DEG_OFFSET )
            self.fname_bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
            self.ssp_file = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/blouin_SSP_coefficients.txt'
            self.bottom_id  ='Coarse-sand'
            
 

