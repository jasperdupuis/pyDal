import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import pandas as pd

# Geographic data and manipulation packages
from geopy import distance,units
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
        x = self.lons_selection
        y = self.lats_selection
        # interpolate.RectBivariateSpline oerates on x,y,z: be explicit here.
        self.interpolation_function = \
            interpolate.RectBivariateSpline(x,
                        y,
                        self.z_selection)
       
        
    def calculate_interp_bathy(self,p_lat_basis,p_lon_basis,p_grid=True):
            x = p_lon_basis
            y = p_lat_basis
            z_interped = self.interpolation_function(
            x,
            y,
            grid=p_grid)
            return z_interped
        # grid = True ==> generates a 2D interpolatin over the xy domain.
        # grid = False ==> generates along the line defined by xy,
        # for single Tx/Rx pair in 2D plane TL(r,z) only, we want False.
    
    def convert_latlon_to_xy_m(self,the_location,lat,lon):
        """
        the_location LAT and LON are nominally CPA.
        """
        R = distance.EARTH_RADIUS * 1000 #km to m
        x = R \
            * self.DEG_TO_RAD * (lon - the_location.LON)\
            * np.cos(np.mean(lat * self.DEG_TO_RAD ) )
        y = R * self.DEG_TO_RAD * (lat- the_location.LAT)

        x = np.array(x)
        y = np.array(y)
        return x,y
    
    
    def get_2d_bathymetry_trimmed(self,
                                  p_location_as_object = 'not set',
                                  p_num_points_lon = 200,
                                  p_num_points_lat = 200,
                                  p_depth_offset = 0):
        """
        stuff.
        """
        self.the_location = p_location_as_object
        self.N_lat_steps = p_num_points_lat
        self.N_lon_steps = p_num_points_lon
        
        
        self.read_bathy(self.the_location.fname_bathy)
        self.sub_select_by_latlon(
            p_lat_extent_tuple = self.the_location.LAT_EXTENT_TUPLE,
            p_lon_extent_tuple = self.the_location.LON_EXTENT_TUPLE) #has default values for NS already
        self.z_selection = self.z_selection - p_depth_offset # z positive ==> below sea level at this point.
        
        self.interpolate_bathy() # assigns interpolation_function
       
    def set_constant_depth(self,p_depth):
        """
        For debugging or Pekeris investigation/comparison,
        take existing bathy function and replace with a constant function.
        """
        constant_depth = np.ones_like(self.z_interped) * p_depth
        self.interpolation_function = \
            interpolate.RectBivariateSpline(
                self.lon_basis_trimmed,
                self.lat_basis_trimmed,
                constant_depth)
         
        
    def plot_bathy(self,
                   p_location,
                   p_N_LAT_PTS,
                   p_N_LON_PTS,
                   p_type = 'extent',
                   p_unit = 'm'):
        """
        p_type = 'extent' for the range extent
        p_type = 'corridor' for just the corridor.
        
        p_unit 'gps' or 'm'
        """
        if p_type == 'extent':
            lat_tuple = p_location.LAT_EXTENT_TUPLE
            lon_tuple = p_location.LON_EXTENT_TUPLE
        if p_type == 'corridor':
            lat_tuple = p_location.LAT_RANGE_CORRIDOR_TUPLE
            lon_tuple = p_location.LON_RANGE_CORRIDOR_TUPLE
        lat_basis = np.linspace(
            lat_tuple[0],
            lat_tuple[1],
            p_N_LAT_PTS)
        lon_basis = np.linspace(
            lon_tuple[0],
            lon_tuple[1],
            p_N_LON_PTS)
        
        self.z_plot = self.calculate_interp_bathy(lat_basis, lon_basis)
        
        if p_unit =='m':
            x,y = self.convert_latlon_to_xy_m (
                p_location,
                lat_basis,
                lon_basis
                )
        if p_unit == 'gps':
            x, y = lon_basis, lat_basis
        
        ext = (
            np.min(x),
            np.max(x),
            np.min(y),
            np.max(y)
            )

        # getting the original colormap using cm.get_cmap() function
        orig_map=plt.cm.get_cmap('viridis')
        # reversing the original colormap using reversed() function
        reversed_map=orig_map
        # reversed_map = orig_map.reversed()
        fig,ax = plt.subplots(1, 1,figsize=(12,7))
        im = ax.imshow(self.z_plot ,
            extent = ext,
            cmap = reversed_map,
            origin = 'lower',
            aspect = 'auto');
        # Include the hydrophone coordinates in the extent image
        if p_type =='extent':    
            lons = np.array(
                [
                p_location.hyd_1_lon,
                p_location.hyd_2_lon
                ]
                )
            lats = np.array(
                [
                p_location.hyd_1_lat,
                p_location.hyd_2_lat
                ]
                )
            if p_unit =='m':
                lons,lats = self.convert_latlon_to_xy_m(p_location, lats, lons)
            ax.scatter(lons,lats,marker='X',color='red')
        plt.colorbar(im)
        return fig,ax


class Bathymetry_CHS_10_100(Bathymetry):
    """
    For interface with CHS NONNA 10 and NONNA 100 data
    Only ASCII format targetted for now, others can be added if wanted in future.
    """
    def read_bathy(self,fname):
        df = pd.read_csv(fname,sep='\t')
        cols = df.columns 
        # index 0: lat, index 1: lon, index 2: depth in m
        lat_deg = df[cols[0]].apply(self.CHS_DMS_to_DecDeg)
        lon_deg = -1 * df[cols[1]].apply(self.CHS_DMS_to_DecDeg) # Canada is western hemisphere only.
        depths = -1 * df[cols[2]] # to match GEBCO format, apply -1 to depth.
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
        z_sel = -sel['Depth (m)'].values 
        
        self.z_selection = z_sel
        self.lats_selection = lat_sel
        self.lons_selection = lon_sel

        del self.df #not needed anymore, release the potentially large memory.

    
    def CHS_DMS_to_DecDeg(self,p_string):
        strs = p_string.split('-')
        d = float(strs[0])
        m = float(strs[1])
        s = float(strs[2][:-2])
        radian = units.radians(degrees=d, arcminutes=m, arcseconds=s)
        deg = radian * 180 / np.pi
        return deg
    
    def interpolate_bathy(self):
        """
        Note this overrides the parent method due to the unstructured data
        provided by CHS.
        
        Need to go from the provided unstructured 1d data from CHS input
        to a well behaved 2d grid, then interpolate over that grid.
        
        Trims the overall arrays to a shape that has no NAN / 0 values
        in the depth array - these would correspond to land. Based
        on magic number n= 0 as a starting point to trim.
        Practically, output is usually with n=2 after looping.
        """
          
        # The entire linear domain captured in the selected data.
        # For unstructured data cannot assume this is OK as-is
        # Must further reduce this after the first interpolation.
        self.lat_basis = np.linspace(np.min(self.lats_selection),
                              np.max(self.lats_selection),
                              self.N_lat_steps)
        self.lon_basis = np.linspace(np.min(self.lons_selection),
                              np.max(self.lons_selection),
                              self.N_lon_steps)
  
        lon_x, lat_y = np.meshgrid(self.lon_basis,self.lat_basis)

        points = (self.lons_selection,self.lats_selection)
        points = np.array(points).T
        self.interpolated_depths = interpolate.griddata(
            (self.lons_selection,self.lats_selection),
            self.z_selection,
            xi=(lon_x,lat_y)
            )

        n = 0 # MAGIC NUMBER #TODO: should be a parameter somewhere
        subarray = self.interpolated_depths
        isNan = np.isnan(np.sum(subarray))
        N_lon, N_lat = self.interpolated_depths.shape[1], \
            self.interpolated_depths.shape[0]
        
        while isNan: # reduce results until no NAN
            n = n + 1
            subarray = self.interpolated_depths[n:N_lat-n,n:N_lon-n]
            isNan = np.isnan(np.sum(subarray))
        self.lon_basis_trimmed = self.lon_basis[n:N_lon-n]
        self.lat_basis_trimmed = self.lat_basis[n:N_lat-n]
        self.z_interped = subarray.T
    
        # Now can assign this as before, with the 2D grid already interpolated over
        # This shouldn't be very different from the interpolated_depths variable above
        # when called to interpolate arbitrary points in the subselected domain.
        # Recall interpolate.RectBivariateSpline explicitly takex it's arguments
        # as x,y,z --> use lon lat, not lat lon.

        self.interpolation_function = \
            interpolate.RectBivariateSpline(
                self.lon_basis_trimmed,
                self.lat_basis_trimmed,
                self.z_interped)
       


class Bathymetry_CHS_2(Bathymetry_CHS_10_100):
    """
    Has a different ASCII format, comma separator not tab.
    """
    def read_bathy(self,fname):
        df = pd.read_csv(fname,sep=',',encoding="UTF-8")
        cols = df.columns 
        # index 0: lat, index 1: lon, index 2: depth in m
        lon_str = cols[0]
        lat_str = cols[1]
        z_str = cols[2]
        lat_deg = df[lat_str].apply(self.CHS_DMS_to_DecDeg)
        lon_deg = -1*df[lon_str].apply(self.CHS_DMS_to_DecDeg)
        depths = df[z_str]
        df[cols[2]] = -1 * df[cols[2]].values #apply negative to have same logic as other formats.
        df['Lats deg'] = lat_deg
        df['Lons deg'] = lon_deg

        self.df = df        
        self.lats = lat_deg
        self.lons = lon_deg
        self.z= depths
    

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
        
# Test the module as top level script
if __name__ == '__main__':
    from locations import Location
    LOCATION = 'Patricia Bay'
    the_location= Location(LOCATION)
    bathy = Bathymetry_CHS_2()
    bathy.get_2d_bathymetry_trimmed( #og variable values
                                  p_location_as_object = the_location,
                                  p_num_points_lon = 200,
                                  p_num_points_lat = 200,
                                  # p_lat_delta = 0.00095,
                                  # p_lon_delta = 0.0015,
                                  p_depth_offset = 0)
    fig_bathy,ax_bathy = \
        bathy.plot_bathy(the_location,
                         p_N_LAT_PTS=48,
                         p_N_LON_PTS=198,
                         )