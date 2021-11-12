import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from netCDF4 import Dataset
import haversine
        
class Bathymetry():
    """
    All bathymetry sources must implement these methods
    
    z    : N x M array of depths (and elevations) over lat and lon bounds
    lats : N x 1 array of latitude coordinates in decimal GPS
    lons : M x 1 array of longitude coordinates in decimal GPS
    
    (var)_selection : trimmed versions of above    
    """
    
    def __init__(self):
        self.z = 'not set'
        self.lats = 'not set'
        self.lons = 'not set'
        
        self.z_selection = 'not set'
        self.lats_selection = 'not set'
        self.lons_selection = 'not set'
        
        self.interpolation_function = r'not set'
        
    def read_bathy(self,fname):
        pass

    
    def show_bathy(self):        
        extents = (min(self.lons),
                   max(self.lons),
                   min(self.lats),
                   max(self.lats))
        plt.imshow(self.z,
                   vmin = -500,
                   vmax = 100,
                   origin = 'lower',
                   extent=extents)
        plt.colorbar()
        plt.xticks=self.lats[::200]
        plt.yticks=self.lats[::200]
        plt.show()

        
    def show_bathy_selection(self):        
        extents = (min(self.lons_selection),
                   max(self.lons_selection),
                   min(self.lats_selection),
                   max(self.lats_selection))
        plt.imshow(self.z_selection,
                   vmin = -100,
                   vmax = 25,
                   origin='lower',
                   extent=extents)
        plt.colorbar()
        plt.xticks=self.lats_selection[::200]
        plt.yticks=self.lons_selection[::200]
        plt.show()


    def interpolate_bathy(self):
        """
        Given the subselection bathymetry, develop an interpolation function over
        the defined space. Then future input lat lon can use this function.
        """
        self.interpolation_function = \
            interpolate.RectBivariateSpline(self.lats_selection,
                        self.lons_selection,
                        self.z_selection)
       
        
    def calculate_interp_bathy(self,p_lat_basis,p_lon_basis,p_grid=False):
            z_interped = self.interpolation_function(
            p_lat_basis,
            p_lon_basis,
            grid=p_grid)
            return z_interped
        # grid = True ==> generates a 2D interpolatin over the xy domain.
        # grid = False ==> generates along the line defined by xy,
        # for single Tx/Rx pair in 2D plane TL(r,z) only, we want False.
    

class Bathymetry_WOD(Bathymetry):
    """
    For interface with WOD provided data.
    """


    def read_bathy(self,fname):
        ds = Dataset(fname)
        
        self.lats = np.array(ds.variables['lat'][:])
        self.lons = np.array(ds.variables['lon'][:])
        self.z= np.array(ds.variables['elevation'][:])
            
    def sub_select_by_latlon(self,
                             p_lat_extent_tuple = (44.4,44.8),
                             p_lon_extent_tuple =  (-63.8,-63.4)):
        """
        slice the existing lat lon z data by defining the lat lon edges
        """
        lat_max_index  = np.argmax((self.lats - p_lat_extent_tuple[1]) > 0)
        lat_min_index= np.argmax((self.lats - p_lat_extent_tuple[0]) > 0)
        lon_min_index = np.argmax((self.lons - p_lon_extent_tuple[0]) > 0)
        lon_max_index = np.argmax((self.lons - p_lon_extent_tuple[1]) > 0)
        
        self.z_selection = self.z[
                                  lat_min_index:lat_max_index,
                                  lon_min_index:lon_max_index]
        self.lats_selection = self.lats[lat_min_index:lat_max_index]
        self.lons_selection = self.lons[lon_min_index:lon_max_index]


class Bathymetry_range_independent_pekeris(Bathymetry_WOD):
    """
    Use the NS Bathymetry to get a basis, then set all values to the depth provided.
    """

    HFX_LAT = 44.64533
    HFX_LON = -63.57239

    def set_depth (self,fname,depth):
        self.read_bathy(fname)
        z = np.ones_like(self.z) * depth
        self.z = z
        
    def set_lat_lon_basis(self,p_distance):
        """
        Taking Halifax as an origin, build a lat lon that has diagonal of provided
        dimension.
        
        dist passed in meters.
        """
        self.hfx_lat_lon = ((44.64533),(-63.57239))
        self.end_lat_lon = haversine.inverse_haversine(((self.HFX_LAT),(self.HFX_LON)),
                                                 p_distance//2,
                                                 haversine.Direction.SOUTHEAST,
                                                 unit=haversine.Unit.METERS)
        self.sub_select_by_latlon(
            (self.hfx_lat_lon[0],self.end_lat_lon[0]),
            (self.hfx_lat_lon[1],self.end_lat_lon[1])
            )
        
