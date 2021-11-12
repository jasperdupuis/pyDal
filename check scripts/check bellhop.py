# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 12:10:36 2021

@author: Jasper
"""
    
import arlpy.uwapm as pm
import arlpy.plot as ac_plt
import numpy as np

env = pm.create_env2d()

pm.print_env(env)
env['nbeams']=40
env['depth']=np.array([[0,20], [300,10], [500,18], [1000,15]])

rays = pm.compute_eigenrays(env)
pm.plot_rays(rays, env=env, width=900)
# compute the arrival structure at the receiver

arrivals = pm.compute_arrivals(env)
pm.plot_arrivals(arrivals, width=900)

ir = pm.arrivals_to_impulse_response(arrivals, fs=96000)
ac_plt.plot(np.abs(ir), fs=96000, width=900)

TL = pm.compute_transmission_loss(env)