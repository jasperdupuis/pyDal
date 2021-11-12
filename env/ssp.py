import numpy as np
import ast


class SSP():
    """
    All SSP sources must map to this class.
    """
    
    class SSP_Error(Exception):
        pass
    
    def __init__(self):
        self.depths = r'not set'
        self.dict = dict()
        self.lat = r'not set'
        self.lon = r'not set'
        self.source = r'not set'
        
    def read_profile(self,fname):
        pass
    
    def get_summer(self):
        return self.dict['Summer']
    
    def get_winter(self):
        return self.dict['Winter']


class SSP_Isovelocity(SSP):
    
    def set_depths(self,p_depths):
        self.depths = p_depths
    
    
    def set_ssp(self,ssp):
        c = np.ones_like(self.depths) * ssp
        self.dict['Summer'] = c
        self.dict['Winter'] = c
    
class SSP_Blouin_2015(SSP):
    """
    Takes as input Stephane Blouin's Bedford Basin coefficients.
    
    Depth-dependent relation only. It can take an arbitrary depth basis function.
    
    The file in which they are has already been edited for this script.    
    
    Third order approximation only.
    """

    def set_depths(self,p_depths):
        self.depths = p_depths
    
    def read_profile(self,
                     fname,
                     summer = r'August',
                     winter = r'February'):
        if self.depths == r'not set':
            raise SSP.SSP_Error("Depths must be set before calling SSP_Blouin.2015.read_profile.")
        self.source = 'Blouin 2015'
        self.lat = 44.693611
        self.lon = -63.640278
        
        with open(fname) as f:
            data = f.readlines()
        
        coeffs_dict = dict()
        for line in data:
            strs = line.split('[')
            coefs = ast.literal_eval('['+strs[1])
            strs[0].split(' ')[0]
            coeffs_dict[strs[0].split(' ')[0]] = coefs

        summer_c = np.zeros(len(self.depths))
        winter_c = np.zeros(len(self.depths))
        for index in range(len(self.depths)):
            summer_c[index] = self.third_order_estimate(
                coeffs_dict[summer],
                self.depths[index])
            winter_c[index] = self.third_order_estimate(
                coeffs_dict[winter],
                self.depths[index])
        self.dict['Summer'] = summer_c
        self.dict['Winter'] = winter_c
            

    def third_order_estimate(self,p_coeffs,p_depth):
        c = 0
        for index in range(len(p_coeffs)):
            c = c + (p_coeffs[index] * p_depth**(index))
        return c

class SSP_Munk(SSP):
    """
    Implement the standard Munk Profile
    
    Must provide a depth vector.
    """
    
    def set_depths(self,p_depths):
        self.depths = p_depths
        self.epsilon = 0.00737
        self.z_scale = 1300 #m/s , typical around 1500m depth.
    
    def read_profile(self,fname='none_required'):
        """
        Hard coded Munk profile generation.
        
        Sets both Winter and Summer profiles to be the same.
        """
        c = []
        for d in self.depths:
            c.append(self.munk(d))
        c = np.array(c)
        self.dict['Summer'] = c
        self.dict['Winter'] = c
        
            
    def munk(self,z):
        z_tilde = 2 * (z - self.z_scale)/self.z_scale
        c = 1500 * (1 + self.epsilon*(z_tilde - 1 + np.exp(-z_tilde)))
        return c
