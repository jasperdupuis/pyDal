# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:27:40 2022

@author: Jasper
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_and_return_2D_axis(
    p_ax,
    p_2d_array,
    p_x_basis,
    p_y_basis,
    p_x_min, #freq value
    p_x_max, #freq value
    p_y_min, #index
    p_y_max, #index
    p_label,
    p_no_f_labels = 6,
    p_no_t_labels = 6,
    p_linear=False):
    """
    trim 2d array to match provided x and y range
    """
    x_min_index = np.argmax((p_x_basis - p_x_min) > 0) + 1
    x_max_index = np.argmax((p_x_basis - p_x_max) > 0) - 1
    y_min_index = p_y_min
    y_max_index = p_y_max
    if y_max_index == 0: y_max_index = -1

    array = p_2d_array[y_min_index:y_max_index, x_min_index:x_max_index]
    x_basis = p_x_basis[x_min_index:x_max_index]
    y_basis = p_y_basis[y_min_index:y_max_index]

    extent = [x_basis[0],x_basis[-1],y_basis[0],y_basis[-1]]
    if p_linear:
        im = p_ax.imshow(array,
                         cmap='RdBu',
                         origin='lower',
                         aspect='auto',
                         extent=extent)
    else:
        im = p_ax.imshow(10 * np.log10(array), 
                         cmap='RdBu',
                         aspect='auto',
                         origin='lower',
                         extent=extent)

    x = x_basis  # gram frequencies
    nx = x.shape[0]
    step_x = int(nx / (p_no_f_labels - 1))  # step between consecutive labels
    # x_positions = np.arange(0, nx, step_x)  # pixel count at label position
    temp_labels = x[::step_x]  # labels you want to see
    x_labels = []
    for l in temp_labels:
        x_labels.append(str(l)[:6])
    #p_ax.set_xticks(x_positions, x_labels)

    y = y_basis  # gram timesteps
    ny = np.array(y).shape[0]
    step_y = int(ny / (p_no_t_labels - 1))  # step between consecutive labels
    # y_positions = np.arange(0, ny, step_y)  # pixel count at label position
    temp_ylabels = y[::step_y]  # labels you want to see
    y_labels = []
    for l in temp_ylabels:
        y_labels.append(str(l)[:4])
    #p_ax.set_yticks(y_positions, y_labels)

    p_ax.set_title(p_label)

    return p_ax,im


def plot_and_return_1D_axis(
    p_ax,
    p_1d_array,
    p_x_basis,
    p_x_min, #freq value
    p_x_max, #freq value
    p_label,
    p_linear=False,
    p_log_x_axis=False):
    """
    trim 2d array to match provided x and y range
    """
    x_min_index = np.argmax((p_x_basis - p_x_min) > 0) + 1
    x_max_index = np.argmax((p_x_basis - p_x_max) > 0) - 1

    array = p_1d_array[x_min_index:x_max_index]
    x_basis = p_x_basis[x_min_index:x_max_index]

    if p_linear:
        im = p_ax.plot(x_basis,array)
    else:
        im = p_ax.plot(x_basis,10 * np.log10(array))

    if p_log_x_axis:
        p_ax.set_xscale('log')
    p_ax.set_title(p_label)

    return p_ax,im