# -*- coding: utf-8 -*-
"""

Coomputes values from literature.

"""

from scipy.stats import skew,kurtosis
import numpy as np


def scintillation_index(x):
    """
    assumes x is a passed time-series of intensity 
    (i.e. a spectral time series for constant f)    
    If analytic function desired, must compute that and pass it.
    
    SI: The intensity variance divided by the square of the
    mean intensity, referred to as the scintillation index
    
    Mudge, 2011 
    Scintillation Index of the Free Space Optical
    Channel: Phase Screen Modelling and Experimental
    Results
    
    and Cote, 2006
    Scintillation index of high frequency acoustic signals forward
    scattered by the ocean surface
    
    and Whitman  1985
    Scintillation index calculations using 
    an altitude-dependent structure constant
    
    Trevorrow, 2021, 
    Examination of Statistics and Modulation of
    Underwater Acoustic Ship Signatures
    
    
    Which in turn references (used this)
    
    Trevorrow, 2004
    STATISTICS OF FLUCTUATIONS IN HIGH-FREQUENCY LOW-GRAZING-ANGLE BACKSCATTER
    
    """
    
    # Mudge complicates this for some reason:
    # # A = hilbert(x) # May not be needed.
    # A = x
    # I_bar = np.mean(A)
    # I2_bar = np.mean(A**2)
    # SI = ( I2_bar ) / ( I_bar ** 2 )
    
    # Cote 2006 definition, classic:
    SI = np.var(x)
    SI = SI / ( np.mean( x ) ** 2 )
    
    return SI
    
def calc_skew(x):
    
    return skew(x)
  
def calc_kurtosis(x):

    return kurtosis(x)


