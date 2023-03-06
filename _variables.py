"""

Control variables accessible across multiple modules as required.

The intent of this is that I Can now just edit this file to change parameters
instead of managing globals across multiple files

"""

from importlib import util

from _imports import np

"""

TRIAL VARIABLES

"""
FS_HYD = 204800
T_HYD = 1.5 #window length in seconds
FS_GPS = 10
LABEL_COM = 'COM '
LABEL_FIN = 'FIN '

"""

LOCATION AND HYDROPHONE SELECTION VARIABLES

"""
HYDROPHONE = 'NORTH'
LOCATION = 'Patricia Bay'

"""

RUN VARIABLES

"""
FREQS = 10 + np.arange(190)
TARGET_FREQ = 73
NUM_DAY = '3' #all results will filter on trial day number.

# An ambient run from July 2019
# TYPE = 'AM'
# MTH = 'J'
# STATE = 'X'
# SPEED='00'        
# HEADING = 'X' #X means both

# A set of dynamic run parameters from July 2019.
TYPE = 'DR'
MTH = 'J'
STATE = 'A'
SPEED='05'        
HEADING = 'X' #X means both

DAY = TYPE + MTH + NUM_DAY #AMJ3, DRF1, etc.

# These are for 0.1 s windows
# INDEX_FREQ_LOW = 1
# INDEX_FREQ_HIGH = 8999 #90k cutoff

# These are for  1.0s windows
INDEX_FREQ_LOW = 3
INDEX_FREQ_HIGH = 89999 #90k cutoff


"""
Tracking of various kinds of lists of runs:
"""

# Runs that need a closer look before they will batch:
OOPS_DYN = ['DRJ3PB09AX02EB',# these runs fail interpolation (needed time basis exceeds provided)
        'DRJ3PB09AX02WB', 
        'DRJ3PB19AX02EB', 
        'DRJ3PB15AX00EN', # There are no track files for these runs.
        'DRJ3PB15AX00WN',
        'DRJ3PB17AX00EN',
        'DRJ3PB17AX00WN',
        'DRJ3PB05AX02EB', # These runs generate hdf5 files with 0 size, but don't fail processing somehow.
        'DRJ2PB11AX01WB',
        'DRJ1PB05BX00WB',
        'DRJ1PB19AX00EB',
        'DRJ1PB05AX00EB', 'DRJ1PB05AX00WB', 'DRJ1PB07AX00EB', #Fucked these up with track overwrite.
        'DRJ1PB07AX00WB', 'DRJ1PB09AX00EB', 'DRJ1PB09AX00WB',
        'DRJ1PB11AX00EB', 'DRJ1PB11AX00WB', 'DRJ1PB13AX00EB',
        'DRJ1PB13AX00WB', 'DRJ1PB15AX00EB', 'DRJ1PB15AX00WB'
        ] 


OOPS_AMB = [ #runs the range fucked up for sure:
        'AMJ1PB00XX00XX',
        'AMJ1PB00XX01XX',
        'AMJ1PB00XX02XX',
        'AMJ1PB00XX04XX',
        'AMJ1PB00XX05XX',
        'AMJ2PB00XX01XX',
        'AMJ2PB00XX02XX',
        'AMJ3PB00XX00XX']

GOOD_AMB = ['AMJ1PB00XX03XX', 
        'AMJ1PB00XX06XX',
        'AMJ1PB00XX07XX',
        'AMJ1PB00XX08XX',
        'AMJ1PB00XX09XX',
        'AMJ1PB00XX10XX',
        'AMJ1PB00XX11XX',
        'AMJ2PB00XX03XX',
        'AMJ3PB00XX01XX',
        'AMJ3PB00XX02XX']

