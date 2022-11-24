import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pandas as pd

# Geographic data and manipulation packages
import geopy
from netCDF4 import Dataset
import haversine
        
from .locations import Location

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
        self.lat_lon_range_steps = 500 # how many steps a range area is chopped in to
        self.DEG_TO_RAD = np.pi / 180
        
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
    
    def convert_latlon_to_xy_m(self,the_location,lat,lon):
        R = geopy.distance.EARTH_RADIUS * 1000 #km to m
        x = R \
            * self.DEG_TO_RAD * (lon - the_location.LON)\
            * np.cos(np.mean(lat * self.DEG_TO_RAD ) )
        y = R * self.DEG_TO_RAD * (lat- the_location.LAT)

        x = np.array(x)
        y = np.array(y)
        return x,y
    
    
    def get_2d_bathymetry_trimmed(self,
                                  p_location_as_string = 'Patricia Bay',
                                  p_num_points_lon = 200,
                                  p_num_points_lat = 50,
                                  p_lat_delta = 0.0015,
                                  p_lon_delta = 0.0015,
                                  p_depth_offset = 0):
        """
        stuff.
        """
        self.the_location = Location(p_location_as_string) 
        
        bathymetry = Bathymetry_CHS()
        bathymetry.read_bathy(self.the_location.fname_bathy)
        bathymetry.sub_select_by_latlon(
            p_lat_extent_tuple = self.the_location.LAT_EXTENT_TUPLE,
            p_lon_extent_tuple = self.the_location.LON_EXTENT_TUPLE) #has default values for NS already
        # If pekeris, set the depth uniformly over the space.
            #APPLY  DEPTH OFFSET - correct for over achieving curve fit and overly granular data
        bathymetry.z_selection = bathymetry.z_selection - p_depth_offset # z negative ==> below sea level at this point.
        
        bathymetry.interpolate_bathy()
        
        self.north_most = self.the_location.LAT + p_lat_delta
        self.south_most = self.the_location.LAT - p_lat_delta
        self.west_most = self.the_location.LON - p_lon_delta
        self.east_most = self.the_location.LON + p_lon_delta
        
        self.lat_basis = np.linspace(self.south_most,
                                     self.north_most,
                                     num=p_num_points_lat)
        self.lon_basis = np.linspace(self.west_most,
                                     self.east_most,
                                     num=p_num_points_lon)
        
        self.z_interped = bathymetry.calculate_interp_bathy(
            self.lat_basis,self.lon_basis,p_grid=True)
        
        
        self.x,self.y = bathymetry.convert_latlon_to_xy_m(
                                                        self.the_location,
                                                        self.lat_basis,
                                                        self.lon_basis)
        self.ext = (np.min(self.x),
                    np.max(self.x),
                    np.min(self.y),
                    np.max(self.y))
    

class Bathymetry_CHS(Bathymetry):
    """
    For interface with CHS NONNA 10 and NONNA 100 data
    Only ASCII format targetted for now, others can be added if wanted in future.
    """
    def read_bathy(self,fname):
        df = pd.read_csv(fname,sep='\t')
        cols = df.columns 
        # index 0: lat, index 1: lon, index 2: depth in m
        lat_deg = df[cols[0]].apply(self.CHS_DMS_to_DecDeg)
        lon_deg = -1*df[cols[1]].apply(self.CHS_DMS_to_DecDeg)
        depths = df[cols[2]]
        df[cols[2]] = -1 * df[cols[2]].values #apply negative to have same logic as other formats.
        df['Lats deg'] = lat_deg
        df['Lons deg'] = lon_deg

        self.df = df        
        self.lats = lat_deg
        self.lons = lon_deg
        self.z= depths
    
    def sub_select_by_latlon(self,
                             p_lat_extent_tuple = (44.4,44.8),
                             p_lon_extent_tuple =  (-63.8,-63.4)):
        """
        slice the existing lat lon z data by defining the lat lon edges
        """
        sel = self.df[ ( self.df ['Lats deg'] > p_lat_extent_tuple[0] ) \
                   & ( self.df ['Lats deg'] < p_lat_extent_tuple[1] ) \
                   & ( self.df ['Lons deg'] > p_lon_extent_tuple[0] ) \
                   & ( self.df ['Lons deg'] < p_lon_extent_tuple[1] ) ]
        lat_sel = sel['Lats deg'].values
        lon_sel = sel['Lons deg'].values
        z_sel = -1*sel['Depth (m)'].values # to match GEBCO format, apply -1 to depth.
        
        self.z_selection = z_sel
        self.lats_selection = lat_sel
        self.lons_selection = lon_sel

        del self.df #not needed anymore, release the potentially large memory.

    
    def CHS_DMS_to_DecDeg(self,p_string):
        strs = p_string.split('-')
        d = float(strs[0])
        m = float(strs[1])
        s = float(strs[2][:-2])
        radian = geopy.units.radians(degrees=d, arcminutes=m, arcseconds=s)
        deg = radian * 180 / np.pi
        return deg
    
    def interpolate_bathy(self):
        """
        Note this overrides the parent method due to the unstructured data
        provided by CHS.
        
        Need to go from the provided unstructured 1d data from CHS input
        to a well behaved 2d grid, then interpolate over that grid.
        """
        # dlat = np.max(self.lats_selection) - np.min(self.lats_selection)
        # dlon = np.max(self.lons_selection) - np.min(self.lons_selection)
        
        # The entire linear domain captured in the selected data.
        # For unstructured data cannot assume this is OK as-is
        # Must further reduce this after the first interpolation.
        lat_basis_whole = np.linspace(np.min(self.lats_selection),
                              np.max(self.lats_selection),
                              self.lat_lon_range_steps)
        lon_basis_whole = np.linspace(np.min(self.lons_selection),
                              np.max(self.lons_selection),
                              self.lat_lon_range_steps)
        # need j * number of steps for np.mgrid parameters:
        stepsj = self.lat_lon_range_steps * 1j 
        self.grid_lon, self.grid_lat = \
            np.mgrid[np.min(lon_basis_whole):np.max(lon_basis_whole):stepsj,\
                     np.min(lat_basis_whole):np.max(lat_basis_whole):stepsj] #the complex indicates number of steps in the interval

        interpolated_depths = interpolate.griddata((self.lons_selection,self.lats_selection),
                                                   self.z_selection,
                                                   xi=(self.grid_lon,self.grid_lat))
        n = 40 # MAGIC NUMBER #TODO
        isNan = True
        N = len(interpolated_depths)
        while isNan:
            n = n + 1
            subarray = interpolated_depths[n:N-n,n:N-n]
            isNan = np.isnan(np.sum(subarray))
        self.lon_basis_trimmed = lon_basis_whole[n:N-n]
        self.lat_basis_trimmed = lat_basis_whole[n:N-n]
    
        
        # Now can assign this as before, with the 2D grid already interpolated over
        # This shouldn't be very different from the interpolated_depths variable above
        # when called to interpolate arbitrary points in the subselected domain.
        self.interpolation_function = \
           interpolate.RectBivariateSpline(self.lat_basis_trimmed,
                       self.lon_basis_trimmed,
                       subarray)


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
        
