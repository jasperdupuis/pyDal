# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:54:58 2021

@author: Jasper
"""

import matplotlib.pyplot as plt 
import numpy as np

from netCDF4 import Dataset

coast = 'NS_range'

if coast == 'BC_range':
    bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/BC_GEBCO/gebco_2021_n52.58056640625001_s46.47216796875001_w-130.5615234375_e-121.00341796875.nc'
    lat_extent = (48.4,48.8)
    lon_extent = (-123.6,-123.2)

if coast =='NS_range':
    bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
    lat_extent = (44.4,44.8)
    lon_extent = (-63.8,-63.4)

if coast =='NS_emerald':
    bathy = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/NS_Bathy/gebco_2021_n52.8662109375_s38.3203125_w-76.376953125_e-50.537109375.nc'
    lat_extent = (44.4,44.8)
    lon_extent = (-63.8,-63.4)


ds = Dataset(bathy)

lats = np.array(ds.variables['lat'][:])
lons = np.array(ds.variables['lon'][:])
z = np.array(ds.variables['elevation'][:])

lat_min_index = np.argmax((lats - lat_extent[0]) > 0)
lat_max_index = np.argmax((lats - lat_extent[1]) > 0)
lon_min_index = np.argmax((lons - lon_extent[0]) > 0)
lon_max_index = np.argmax((lons - lon_extent[1]) > 0)

z_sliced = z[lat_min_index:lat_max_index,lon_min_index:lon_max_index]

extents = (lons[lon_min_index],
           lons[lon_max_index],
           lats[lat_min_index],
           lats[lat_max_index])
plt.imshow(z_sliced,
           vmin = -500,
           vmax = 100,
           origin='lower',
           extent=extents)
plt.colorbar()
plt.xticks=lons[lon_min_index:lon_max_index:200]
plt.yticks=lats[lat_min_index:lat_max_index:200]

