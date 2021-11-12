import numpy as np


class SeaBed():
    """
    Data about the sea floor.
    
    init requires bathy to be already loaded.
    """    
  
    def __init__(self,bathy_object):
        self.lats_selection = bathy_object.lats_selection
        self.lons_selection = bathy_object.lons_selection
        self.bottom_type_profile = np.ones_like(bathy_object.z_selection) 

    def read_default_dictionary(self,
                                fname = r'C:/Users/Jasper/Desktop/MASC/Environmental Data/seabed_numbers_from_saleh_rabah.txt'):
        dictionary = dict()
        with open(fname) as f:
            lines = f.readlines()
            for line in lines:
                strs = line.split(' ')
                temp = dict()
                temp['M_z'] = float(strs[-4])
                temp['Rho'] = float(strs[-3])
                temp['c'] = float(strs[-2])*1000         #CONVERT km/s to m/s
                temp['alpha'] = float(strs[-1])
                dictionary[strs[0]] = temp
        self.sediment_dictionary = dictionary
        
        
    def assign_single_bottom_type(self,p_type = r'Sand-silt'):
        temp = self.bottom_type_profile #has dimensinos of bathymetry in constructor
        self.bottom_type_profile = dict()
        for key,value in self.sediment_dictionary[p_type].items():
            self.bottom_type_profile[key] = temp * value
        