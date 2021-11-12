import numpy as np

class Surface():
    """
    Surface geometry and, later on, scattering parameters (Bubbles)
    """
    def __init__(self):
        return
    
    def SS_0(self,distance):
        surface = np.array([[r, 01.+0.1*np.sin(2*np.pi*0.005*r)] for r in np.linspace(0,distance,25)])
        self.surface_desc = surface
