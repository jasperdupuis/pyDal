import matplotlib.pyplot as plt # for visualazing during testing
import torch
import numpy as np
import time
import pickle
from scipy import interpolate
import random

# BELLHOP interface
import arlpy.uwapm as pm

# my modules
import sys
sys.path.insert(1,r'C:\Users\Jasper\Documents\Repo\pyDal')
from pyDal.env.environment import Environment_ARL # only bellhop for synthetic data for now...
from pyDal.env.locations import Location
from pyDal.env.bathymetry import Bathymetry_WOD
from pyDal.env.ssp import SSP_Blouin_2015, SSP_Munk, SSP_Isovelocity
from pyDal.env.seabed import SeaBed
from pyDal.env.surface import Surface

NUM_SYNTH_RUNS = 20

M_PER_S_TO_KNOTS = 1.94 # 1 m/s is 1.94 knots
DEG_PER_RAD = 180/np.pi
GPS_FREQ = 10 #Hz
AVG_EARTH_RADIUS_M = 6378000 # good enough for government work.

default_run_dictionary = {
    'RL_f' : 120.,
    'noise_power' : 1.0, #dB
    'nominal_length' : 200., #m
    'nominal_SOG' : 9., # kts
    'cpa_offset' : 5., # m
    'track_angle_deg' : 0.# degrees
    }

default_TL_model_dictionary = {
    'LOC_STRING' : "Ferguson Cove",
    'DEPTH_OFFSET' : 0., # m, can help protect against overly granular datasets / overfitting
    'BASIS_SIZE_depth' : 200, #m
    'BASIS_SIZE_distance' : 200, #m
    'PEKERIS_DEPTH' : 100., #m
    'N_BEAMS' : 100., # 
    'KRAKEN_ROUGHNESS' : [0.1,0.1], # m
    'RAM_DELTA_R' : 1., # m, step size
    'RAM_DELTA_Z' : 1., # m, step size
    'HYD_DYN_EAST_DEPTH' : 27.6,
    'HYD_DYN_EAST_LAT' :44.60551039,
    'HYD_DYN_EAST_LON' : -63.54031211,
    'HYD_DYN_WEST_DEPTH' : 27.2,
    'HYD_DYN_WEST_LAT' : 44.60516000,
    'HYD_DYN_WEST_LON' : -63.54272550
    }


BASIS_SIZE_depth = 200
BASIS_SIZE_distance = 200
PEKERIS_DEPTH = 100
N_BEAMS = 100 #this doesn't appear to matter - bellhop parameter
KRAKEN_ROUGHNESS = [0.1,0.1] # Doesn't change anything apparently
RAM_DELTA_R = 1 # m, range step size
RAM_DELTA_Z = 1 # m, depth step size, not currently used by me (Default is computed in pyram)
   


class Synthetic_f_xy_dataset(torch.utils.data.Dataset):
    """
    Developed for TL synthetic data set over an x,y space. This is only for the
    TL "equalizer" investigation at this point - 20220916
    
    Two methods of inputing x,y,TL data are possible:
            1)  uniform distribution for basic testing.
    
                i) input a x_range and y_range as arrays of possible values to 
                draw random samples from. Do this using create_sample_coordinates.
                ii) then, build the set of labels (outside this class), with 
                values corresponding to the synthetic TL for that x,y coordinate.
                iii) the result is a UNIFORMLY distrubted sample over the space defined 
                by x_range, y_range. Use case code is included as comments to create_sample_coordinates.
                
            2)  Simulated ship track data with labels for more realistic testing.
                i)  define track angle through CPA, and CPA offset.
                ii) 

    Cartesian representation has negative values at the bottom.
    "Bottom"  would correspond to "last" indexing a 2D array.
    But, by default np.arange behaviour, negative numbers are "first".
    Therefore y_range is multiplied by -1 for indexing reasons.
    
    x_range does not need this treatment; negative values are "first" 
    (left-most )in both program and cartesian sense.
    """
    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Model inputs and labels are being moved to {self.device} device.\n\n")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        x and y are here inputs and labels, not cartesian
        """
        x = self.inputs[idx,:]
        y = self.labels[idx]
        x = x.float()
        y = y.float()
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y

    def create_sample_coordinates(self,x_range,y_range,n=10000):
        """
        from self.x_range and self.y_range, create input tensor for training.
        
        Use this with draw_labels_from_coordinates after.
        
        You must check x, y coordinates make sense first!
        (i.e. reconcile your cartesian vs indexing conventions)
        
        USe case code for this function:
            #Set up the cartesian geometry
        xmin = -10
        xmax = 10
        ymin = -50
        ymax = 50
    
        x_range = np.arange(xmin,xmax)
        y_range = np.arange(ymin,ymax)
        x_size = xmax-xmin
        y_size = ymax-ymin
        x_surface = np.ones((y_size,x_size)) # dim1 is column index, dim2 is row index
        y_surface = (x_surface[:,:].T * np.arange(ymin,ymax)*-1).T # hackery to use the numpy functions, no big deal
        x_surface = x_surface[:,:] * np.arange(xmin,xmax)
        
        n_train = 50000
        # INSTANTIATE THE OBJECT 
        dset_train = Synthetic_f_xy_dataset(x_range, y_range)
        dset_train.create_sample_coordinates(int(n_train))
        dset_train.draw_labels_from_coordinates(label)
        """
        self.x_range = x_range
        self.y_range = y_range

        x = np.random.choice(self.x_range,size = n)
        y = np.random.choice(self.y_range,size = n)
        x = torch.tensor(x) # x coordinate is column index!
        y = torch.tensor(y) # y coordinate is row index!
        self.y = y
        self.x = x
        self.inputs = torch.column_stack((y,x))
        
    def draw_labels_from_coordinates(self,labels):
        """
        labels is an array of dimensions len(y_range),len(x_range)
        for N data samples (set outside class, passed to create_sample_coordinates),
        self.inputs is N x 2 tensor corresponding to REAL y,x coordinates.
        [Note y, x not x,y (row, col indexing, not cartesian) ]

        The real y,x values must be convered to index values in order to 
        retrieving the label value associated with the real y,x pair.
        
        use self.x_range and self.y_range to find these indices - they are unique sets.        
        
        use case for this function:
            
            # Y variation in synthetic data
        fy = 6*np.sin(4*np.pi*y_surface/y_size)
        y_slope = 0.00
        fy = fy + y_surface * y_slope
            # X component of synthetic data
        fx = 6*np.sin(4*np.pi*x_surface/x_size)
        fx = np.zeros_like(x_surface)
        x_slope = 0.
        x_slope = 0.2
        fx = fx + x_surface * x_slope 
        
        source_level = np.ones((y_size,x_size)) * RL_f
        noise = noise_power*np.random.rand(y_size,x_size) - noise_power/2
        source_level = source_level + noise
        received_level = source_level - (fx+fy) # x and y variation
        label = received_level - np.max(received_level) 

        
        """
        coords_temp = np.array(self.inputs)
        res_temp = np.zeros(self.__len__())
        for index in range(self.__len__()):
            coord = coords_temp[index,:]
            ind_row = np.where(self.y_range == coord[0]) # y
            ind_col = np.where(self.x_range == coord[1]) # x
            res_temp[index] = labels[ind_row,ind_col]
        self.labels = torch.tensor(res_temp)
        
    def load_xy_and_label_data(self,x,y,labels):
        """
        A simple function which assumes x,y,labels are good. i.e.
        already flattened and rationalized/checked for accuracy (no mixed up indices)
        Should be tensor inputs, np.ndarry also OK. nx1 only.
        
        To farm out the generation of the nx1 inputs from a set of iterables,
        (e.g. from a three lists of "real" run data), see "concat_xy_label_data"
        """
        self.y = y
        self.x = x
        x = torch.tensor(x)
        y = torch.tensor(y)
        self.inputs = torch.column_stack((y,x))
        self.labels = torch.tensor(labels)
        
    def concat_and_load_xy_label_data(self,p_dictionary):
        """
        
        p_dictionary has keys (not used here), where each value of 
        p_dictionary[key] is another dictionary with keys:
            'X'
            'Y'
            'TL_label'
        all of which provide a 1-d array of same length.
        
        Unpack this structure and load using load_xy_and_label_data
        """
        # Find the length of result vectors
        count = 0
        for key,value in p_dictionary.items():
            count = \
                count + \
                len(value['X'])
        
        # Allocate memory                
        xx = np.zeros(count,dtype=np.float64)
        yy = np.zeros_like(xx)
        ll = np.zeros_like(xx)        
        
        # Assign values to memory
        count = 0 # for indexing results arrays
        for key,value in p_dictionary.items():
            n = len(value['X'])
            xx[count:count + n] = value['X']
            yy[count:count + n] = value['Y']
            ll[count:count + n] = value['TL_label']
            count = count + n # do this after using count to assign memory 
            # for this iteration
        
        self.load_xy_and_label_data(xx, yy, ll)
            
        
    def generate_interpolation_function_TL(self,x,y,TL):
        """
        Given the subselection bathymetry, develop an interpolation function over
        the defined space. Then future input x y can use this function.
        
        20220922 Not totally satistified x,y are called in the right order 
        on RectBivariateSpline. For old lat-lon work this was in lat-lon order 
        i.e. y,x.
        
        x,y must both be strictly ascending order before the call to 
        RectBivariateSpline, so address this in the function.
        """
        dx = np.abs(x[1] - x[0])
        dy = np.abs(y[1] - y[0])

        x = np.arange(np.min(x),np.max(x)+1,dx)
        y = np.arange(np.min(y),np.max(y)+1,dy)
        
        self.interpolation_function = \
            interpolate.RectBivariateSpline(
                        y,
                        x,
                        TL)
            
    def calculate_interp_TL(self,p_x_basis,p_y_basis,p_grid=False):
        """
        # grid = True ==> generates a 2D interpolatin over the xy domain.
        # grid = False ==> generates along the line defined by xy,
        # for single Tx/Rx pair in 2D plane TL(r,z) only, we want False.

        """
        TL_interped = self.interpolation_function(
        p_y_basis,
        p_x_basis,
        grid=p_grid)
        return TL_interped
        
    @staticmethod
    def build_dset_with_n_random_tracks_with_TL(
            p_fname = r'C:/Users/Jasper/Documents/Repo/pyDal/synthetic-data-sets/synthetic_TL/synthetic_TL_Bellhop_1000.pkl',
            p_num_run = 20,
            p_std_angle = 2,
            p_std_SOG = 0.2,
            p_std_CPA = 4):
        """
        For synthetic data only, real data needs less work
        
        Nests several functions in this module and class
        so that other modules can get this pretty easily.
        
        Requires an absolute pickled dictionary path.

        1) Makes interpolation function over the pickled TL(x,y) array
            The pickled dictionary needs keys of:
                'X', 'Y', 'TL_cmplx'
                Where X, Y are returned in meters
                'TL_cmplx' is the complex array from the Bellhop calculation
                Must apply 20log10(abs) to TL_cmplx
        2) makes a synthetic dictionary with p_num_run entries
            each run has a sub dictionary with keys
                'X', 'Y', 'TL_label' 
                where these are numpy arrays
        3) stack X,Y, TL for torch use
            Using a class instance method take the synthetic data dict
            and unzip in to 3x 1-d numpy arrays, then cast to torch.tensor
            then return dataset for training/validation.


        Returns
        -------
        dataset object with x, y, label already loaded for training.

        """
        with open(p_fname, 'rb') as f:
            result_dictionary = pickle.load(f)    
        # From dictionary of X, Y, TL_cmplx, treat NAN and INF values
        # And then make interpolation function
        # This isn't necessary for real data (would be x, y, RL(f))
        # Interpolation not necessary for real data!    
        TL = 20*np.log10(np.abs(result_dictionary['TL_cmplx']))
        TL = Synthetic_f_xy_dataset.interpolate_nan_adjacent_means(TL)
        x_range  = result_dictionary['X']
        y_range = result_dictionary['Y']
        dataset = Synthetic_f_xy_dataset()
        dataset.generate_interpolation_function_TL(x_range, y_range, TL)
        
        # 
        synth_data_dict = dict()
        for index in range(NUM_SYNTH_RUNS):
            run_dict = dict(default_run_dictionary) # copy of default
            run_dict = \
                Synthetic_f_xy_dataset.randomize_run_parameters_normal(
                    run_dict,
                    p_std_SOG,
                    p_std_angle,
                    p_std_CPA,)
            run_dict = Synthetic_f_xy_dataset.complete_run_dictionary(run_dict)    
            x_track,y_track = x,y = Synthetic_f_xy_dataset.build_synthetic_track(run_dict)
            synth_labels = dataset.calculate_interp_TL(x_track,y_track,p_grid=False) #see func defn
            temp = dict()
            temp['X'] = x_track
            temp['Y'] = y_track
            temp['TL_label'] = synth_labels
            synth_data_dict[index] = temp
    
        dataset.concat_and_load_xy_label_data(synth_data_dict)
        
        return dataset

    @staticmethod
    def randomize_run_parameters_normal(p_run_dict,
                                 p_nominal_SOG_std = 0.2,
                                 p_nominal_angle_std = 2,
                                 p_cpa_offset_std = 4):
        """
        Normal distribution of key run parameters, defaults should be
        close to good values.
        returns p_run_dict (which has nominal values), with the
        randomization applied attributable to other parameters.
        """        
        dSOG = random.gauss(0, p_nominal_SOG_std)
        dANGLE= random.gauss(0, p_nominal_angle_std)
        dCPA = random.gauss(0, p_cpa_offset_std)
        p_run_dict['nominal_SOG'] = p_run_dict['nominal_SOG'] + dSOG
        p_run_dict['track_angle_deg'] = p_run_dict['track_angle_deg'] + dANGLE
        p_run_dict['cpa_offset'] = p_run_dict['cpa_offset'] + dCPA
        return p_run_dict        
        
    @staticmethod
    def complete_run_dictionary(dictionary = default_run_dictionary):
        """
        given a dictionary with key run parameters,
        complete the dictionary for additional processing.
        """
        dictionary['nominal_mps'] = dictionary['nominal_SOG']\
                                /M_PER_S_TO_KNOTS
        dictionary['track_angle_rad'] = dictionary['track_angle_deg']\
                                /DEG_PER_RAD
        dictionary['seconds'] = dictionary['nominal_length']\
                                / dictionary['nominal_mps']
        dictionary['n_points'] = int ( dictionary['seconds']\
                                      * GPS_FREQ )
        return dictionary


    @staticmethod
    def build_synthetic_track(dictionary):
        """
        Cartesian +ve angle ==> CCW from +ve x axis
        Geographic +ve angle ==> CW from North
        ==> must apply -1 to the rotation angle here.
        """
        indices= np.arange(dictionary['n_points'])
        x_coord = np.zeros_like(indices)
        y_coord = indices / GPS_FREQ #time samples now
        y_coord = y_coord * dictionary['nominal_mps']
        y_coord = y_coord - np.mean(y_coord)
  
        x_coord = np.cos(-dictionary['track_angle_rad']) * x_coord \
            - np.sin(-dictionary['track_angle_rad']) * y_coord
        y_coord = np.sin(-dictionary['track_angle_rad']) * x_coord \
            + np.cos(-dictionary['track_angle_rad']) * y_coord
        
        x_coord = x_coord + dictionary['cpa_offset']
        return x_coord, y_coord
        
    
    @staticmethod
    def build_lat_lon_approximation(dx = 4, #0.25m resolution
                                    dy = 4, # 0.25m resolution
                                    x_range = (-25,25), #(westernmost, easternmost)
                                    y_range = (-100,100), #(southernmost, northernmost)
                                    cpa_lon = -63.541518805, #ferguson cove default
                                    cpa_lat = 44.605335194999995):#ferguson cove default
        """
        For distances typical of acoustic ranging, we can assume a flat earth.
        So find the range nominal CPA, and build the lat-lon range
        from desired X-Y size in meters.
        
        x_range = (min,max), #(westernmost, easternmost)
        y_range = (min,max), #(southernmost, northernmost)
        
        ( north + west hemisphere assumption )
        
        """
        sx = np.arange(x_range[0],x_range[1], 1 / dx )
        sy = np.arange(y_range[1],y_range[0], -1 / dy) # note order due to 
        # north/south opposite y axis and index convention.
        
        lats = ( sy / AVG_EARTH_RADIUS_M ) #result of this operation is in rads
        lats = lats * DEG_PER_RAD
        lats = lats + cpa_lat 
        lons =  sx / ( np.cos(cpa_lat/DEG_PER_RAD) * AVG_EARTH_RADIUS_M ) #result of this operation is in rads
        lons = lons * DEG_PER_RAD
        lons = lons + cpa_lon
        
        return lats,lons,sx,sy

    @staticmethod
    def Calculate_TL_Bellhop_from_TX_and_RX(
          freq,
          tx_lats, # a vector of each value - loop over these.
          tx_lons, # A vector of each value - loop over these.
          rx_hyd_depth,
          rx_lat,
          rx_lon,
          location_string = default_TL_model_dictionary['LOC_STRING'],
          p_source_depth = 4, #default for propeller center depth
          p_basis_size_depth = default_TL_model_dictionary['BASIS_SIZE_depth'],
          p_basis_size_range = default_TL_model_dictionary['BASIS_SIZE_distance'],
          p_depth_offset = default_TL_model_dictionary['DEPTH_OFFSET'], 
          ):
        """
        This is fucking ugly, but produces some test data in ~ 25 minutes 
        for a single freq (!)
        
        Build the ARL Bellhop environment from previous work, then loop over
        the set of lat-lon coordinates to build a TL array. There is indexing
        to track the lat-lon pairs with each TL calculation.
        
        To rebuild a surface representation need to unpack this result.
        
        Sample code to call this as of 20220922 (see also non-class method
         build_and_save_TL_model() in this file)
        
        #Create the range corridor in meters (default is -25,25, -100,100)
        lats_input, lons_input = 
            Synthetic_f_xy_dataset.build_lat_lon_approximation(dx=1,dy=1)
        # get the hydrophone lat and lon and set the depth.
        rx_lat = default_TL_model_dictionary['HYD_DYN_EAST_LAT']
        rx_lon = default_TL_model_dictionary['HYD_DYN_EAST_LON']
        # inperpolation from bathy DB inaccurate depth at hydrophone, 
        # this is closest whole number
        rx_d = 18. 
        # Set the freq of interest
        freq = 1000.
        
        print("there are " +str(len(lats_input) ) + " lats to compute")
        #do the computation and time it.
        # for dx, dy = 1 and -25,25 and -100,100, this took ~ 25 minutes
        start = time.time()        
        lats, lons, TLs_cmplx, R = 
            Synthetic_f_xy_dataset.Calculate_TL_Bellhop_from_TX_and_RX(
            freq,
            lats_input,
            lons_input,
            rx_d,
            rx_lat,
            rx_lon
            )
        end = time.time()
        print(end-start)
    
        results = dict()
        results['lats'] = lats
        results['lons'] = lons
        results['TL_cmplx'] = TLs_cmplx
        results['R'] = R

        # Store dictionary of results to file, note frequency is NOT in 
        # filename here ! 
        # So don't overwrite it!
        
        with open('synthetic_TL_Bellhop.pkl', 'wb') as f:
            pickle.dump(results, f)
    
        # Retrieval
        # with open('synthetic_TL_Bellhop.pkl', 'rb') as f:
        #     results = pickle.load(f)    
        
        """
          
        the_location = Location(location_string)
    
        bathymetry = Bathymetry_WOD()
        bathymetry.read_bathy(the_location.fname_bathy)
        bathymetry.sub_select_by_latlon(
            p_lat_extent_tuple = the_location.LAT_EXTENT_TUPLE,
            p_lon_extent_tuple = the_location.LON_EXTENT_TUPLE) #has default values for NS already
        
        #APPLY  DEPTH OFFSET - correct for over achieving curve fit and overly granular data
        bathymetry.z_selection = bathymetry.z_selection - p_depth_offset # z negative ==> below sea level at this point.
    
        bathymetry.interpolate_bathy()
        
        surface = Surface()
        bottom_profile = SeaBed(bathymetry)
        bottom_profile.read_default_dictionary()
        bottom_profile.assign_single_bottom_type(the_location.bottom_id)

        TLs_cmplx = np.zeros((len(tx_lats), len(tx_lons)),dtype = np.complex128)
        R = np.zeros_like(TLs_cmplx,dtype = float)
        lats = np.zeros_like(TLs_cmplx,dtype = float)                
        lons = np.zeros_like(TLs_cmplx,dtype = float)
        for lat_index in range(len(tx_lats)):
            for lon_index in range(len(tx_lons)):
                # for start/end lat-lons, this work concerned only with western
                # and northern hemisphere. neglect the other cases.
                #   for lattitude, we wish to start with the northernmost value
                # This corresponds with the 0th row of the resulting array
                lat_start = max(tx_lats[lat_index],rx_lat)
                lat_end = min(tx_lats[lat_index],rx_lat)
                #   for longitude, we wish to start with the westernmost value.
                # corresponds with 0th row of resulting array
                lon_start = min(tx_lons[lon_index],rx_lon)
                lon_end = max(tx_lons[lon_index],rx_lon)
                
                lat_basis = np.linspace(lat_start,lat_end,num=BASIS_SIZE_distance)
                lon_basis = np.linspace(lon_start,lon_end,num=BASIS_SIZE_distance)
                z_interped = bathymetry.calculate_interp_bathy(lat_basis,lon_basis)
            
                # THIS IS KIND OF OUT OF ORDER, BUT NEED IT HERE FOR KOSHER SSP WITH KRAKEN
                MAX_LOCAL_DEPTH = np.abs(np.min(z_interped))
                MAX_LOCAL_DEPTH +=10 # THIS IS A HACK TO MAKE SURE SSP EXTENDS PAST BOTTOM.
                
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
                
                
                
                env_ARL = Environment_ARL()
                env_ARL.set_bathymetry(bathymetry)
                env_ARL.set_seabed(bottom_profile)
                env_ARL.set_ssp(ssp)
                env_ARL.set_surface(surface)
        
        
                env_bellhop = env_ARL.create_environment_model(
                    (rx_lat,rx_lon),
                    (tx_lats[lat_index],tx_lons[lon_index]),
                    FREQ_TO_RUN = freq,
                    RX_HYD_DEPTH = rx_hyd_depth,
                    TX_DEPTH = p_source_depth,
                    N_BEAMS = N_BEAMS,
                    BASIS_SIZE_DEPTH = p_basis_size_depth,
                    BASIS_SIZE_DISTANCE = p_basis_size_range,
                    )
                
                TL = pm.compute_transmission_loss(
                env_bellhop,
                mode=pm.coherent,
                )
    
                try:
                    x_cmplx = TL.iloc(0)[0].iloc(0)[0]
                    TLs_cmplx[lat_index,lon_index] = x_cmplx
                    R[lat_index,lon_index] = env_bellhop['rx_range']
                except:
                    print('Error at range: ' +str(env_bellhop['rx_range']))
                    
                lats[lat_index,lon_index] = tx_lats[lat_index]
                lons[lat_index,lon_index] = tx_lons[lon_index]

        return lats, lons, TLs_cmplx, R    

    @staticmethod 
    def interpolate_nan_adjacent_means(array):
        """
        Given a 2D array, identify all non-finite entries (nan or inf)
        and then fill those indices with the average of the four immediately
        adjacent cells.
        
        Average is applied in passed domain directly - doesn't know about dB

        """    
        # smooth the result to interpolate nan or inf results
        # this will only work for cases where all orthogonal cells from inf/nan
        # are themselves NOT inf or nan.
        result = array
        infs = np.isfinite(result) == False 
        # #bool array, True ==> value is nan or inf
        while np.sum(infs ) > 0: #sum over the entire array TL_infs
            index_row = np.where(infs )[0][0] # first dimension
            index_col = np.where(infs )[1][0] # second dimension
            value = result[index_row - 1 , index_col] + result[index_row + 1 , index_col] \
                + result[index_row , index_col - 1 ] + result[index_row , index_col + 1] 
            result[index_row,index_col] = value/4 # insert the mean of adjacent cells
            infs = np.isfinite(result) == False 
        return result              



def build_and_save_TL_Bellhop_model(p_freq=1000,
                            p_rx_d = 18.,
                            p_dx = 1,
                            p_dy = 1):
    """
    Pretty cool to see the result!
    Something like, if local to this function:
    import matplotlib.pyplot as plt
    plt.imshow(20*np.log10(np.abs(TLs_cmplx)));plt.colorbar()
    """
    lats_input, lons_input,xs,ys = Synthetic_f_xy_dataset.build_lat_lon_approximation(dx=p_dx,dy=p_dy)
    
    rx_lat = default_TL_model_dictionary['HYD_DYN_EAST_LAT']
    rx_lon = default_TL_model_dictionary['HYD_DYN_EAST_LON']
    rx_d = p_rx_d # inperpolation from bathy DB inaccurate depth at hydrophone, this is closest whole number
    freq = p_freq
    
    print("there are " +str(len(lats_input) ) + " lats to compute")
    #do the computation and time it.
    # for dx, dy = 1 and -25,25 and -100,100, this took ~ 25 minutes
    start = time.time()        
    lats, lons, TLs_cmplx, R = Synthetic_f_xy_dataset.Calculate_TL_Bellhop_from_TX_and_RX(
        freq,
        lats_input,
        lons_input,
        rx_d,
        rx_lat,
        rx_lon        
        )
    end = time.time()
    print(end-start)

    results = dict()
    results['lats'] = lats
    results['lons'] = lons
    results['X'] = xs
    results['Y'] = ys
    results['TL_cmplx'] = TLs_cmplx
    results['R'] = R

    fname = r'synthetic_TL/synthetic_TL_Bellhop_' + str(p_freq).zfill(4) + r'.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(results, f)


def build_and_save_TL_sine_model():
    """
    Holds early work from the x-y sinusoidal variation days.

    20220922: Not really used anymore, uses magic numbers everywhere.
    """
    # Chose a frequency value of interest with "true" RL value
    f = 1000
    RL_f = 120 #dB ref uPa^2 / Hz    
    # Assign a noise power value
    noise_power = 1.0 #dB
    
    #Set up the cartesian geometry
    xmin = -25
    xmax = 25
    ymin = -100
    ymax = 100

    x_range = np.arange(xmin,xmax)
    y_range = np.arange(ymin,ymax)
    x_size = xmax-xmin
    y_size = ymax-ymin
    x_surface = np.ones((y_size,x_size)) # dim1 is column index, dim2 is row index
    y_surface = (x_surface[:,:].T * np.arange(ymin,ymax)*-1).T # hackery to use the numpy functions, no big deal
    x_surface = x_surface[:,:] * np.arange(xmin,xmax)
    
    # Define a synthetic function over the corridor to apply TL
    # Easiest is x- and y- dependent separately.
    
    # Y component of synthetic data.
    fy = 6*np.sin(4*np.pi*y_surface/y_size)
    # fy = np.zeros_like(y_surface)
    y_slope = 0.00
    fy = fy + y_surface * y_slope
    # X component of synthetic data
    fx = 6*np.sin(4*np.pi*x_surface/x_size)
    fx = np.zeros_like(x_surface)
    x_slope = 0.
    x_slope = 0.2
    fx = fx + x_surface * x_slope 
    
    source_level = np.ones((y_size,x_size)) * RL_f
    noise = noise_power*np.random.rand(y_size,x_size) - noise_power/2
    source_level = source_level + noise
    received_level = source_level - (fx+fy) # x slope and y cos
    # received_level = source_level - (fx) # just x slope
    # received_level = source_level - (fy) # just y curve slope
    label = received_level - np.max(received_level)
    return label


def create_synthetic_uniform_datasets(fname,
                                      n_train = 40000,
                                      n_val = 5000):
    """
    Allow uniform sampling from already-created & pickled TL(x,y) arrays
    
    fname must be fully qualified absolute path
    """
    
    with open(fname, 'rb') as f:
        result_dictionary = pickle.load(f)    
    
    # train data
    TL = 20*np.log10(np.abs(result_dictionary['TL_cmplx']))
    TL = Synthetic_f_xy_dataset.interpolate_nan_adjacent_means(TL)
    label = TL
    x_range  = result_dictionary['X']
    y_range = result_dictionary['Y']

    #Set up the cartesian geometry from dictionary
    xmin = np.min(x_range)
    xmax = np.max(x_range) + 1
    ymin = np.min(y_range)
    ymax = np.max(y_range) + 1

    x_range = np.arange(xmin,xmax)
    y_range = np.arange(ymax,ymin,-1)
    x_size = int(xmax-xmin)
    y_size = int(ymax-ymin)
    x_surface = np.ones((y_size,x_size)) # dim1 is column index, dim2 is row index
    y_surface = (x_surface[:,:].T * y_range).T # hackery to use the numpy functions, no big deal
    x_surface = x_surface[:,:] * np.arange(xmin,xmax)
    
    dset_train = Synthetic_f_xy_dataset()
    dset_train.create_sample_coordinates(x_range,y_range,int(n_train))
    dset_train.draw_labels_from_coordinates(label)

    dset_val = Synthetic_f_xy_dataset()
    dset_val.create_sample_coordinates(x_range,y_range,int(n_val))
    dset_val.draw_labels_from_coordinates(label)

    return dset_train, dset_val, label, x_surface, y_surface


if __name__ == "__main__":
    
    # build_and_save_TL_Bellhop_model(50)
    
    # open local file name to this module for testing,
    # dictionary pickled after build_and_save_TL_Bellhop_model(freq)
    fname = 'C:/Users/Jasper/Documents/Repo/pyDal/synthetic-data-sets/synthetic_TL/synthetic_TL_Bellhop_0050.pkl'
    
    dataset = Synthetic_f_xy_dataset.build_dset_with_n_random_tracks_with_TL(
        fname,
        p_num_run = 20,
        p_std_angle = 2,
        p_std_SOG = 0.2,
        p_std_CPA = 4)

