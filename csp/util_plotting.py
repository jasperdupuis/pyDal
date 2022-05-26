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
    y_min_index = np.argmax((p_y_basis - p_y_min) > 0) + 1
    y_max_index = np.argmax((p_y_basis - p_y_max) > 0) - 1
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


def plot_multi_data(
        p_data_dict,
        p_identifiers, # identifiers that exist in passed p_data_Dict
        p_data_ref, #strings that match columns in p_data_dict
        p_x_ref,#string
        p_y_ref,#string
        p_shape,#[nrow,ncol], integers.
        p_linear=False,
        p_subheading = '', #string
        p_units = 'arbitrary', #string
        p_xlims = [0,0], #[min,max], array extents.
        p_ylims = [0,0] #[min,max], scale extents.
                    ):
    """

    Parameters
    ----------
    p_data_dict : dict()
        dictionary with structure :
                head
                  - identifier_1
                    -2d data array
                    -x axis data
                    -y axis data
        where the string accessors match p_data_ref,p_x_ref,p_yref.
                  
    p_identifiers : [string]
        Unique identifiers that exist in passed p_data_Dict        
    p_data_ref : string
        String that match data key in p_data_dict        
    p_x_ref : string
        String that match x-axis key in p_data_dict        
    p_y_ref : string
        String that match y-axis key in p_data_dict        
    p_shape : tuple (2-tuple)
        Ordered tuple that provides nrow, ncol in final figure.
    p_subheading : string, optional
        Option argument adds sub-heading to main figure title
        for extra information (on a second line)
    p_units : string, optional
        The default is 'arbitrary'.
    p_xlims : tuple (2-tuple), optional
        Provide the float axis min and max values. If these two are equal (default),
        then will get the whole range. Subcalls prune the indices.
    p_ylims : tuple (2-tuple), optional
        Provide the float min and max values. If these two are equal (default),
        then will get the whole range. Subcalls prune the indices.
    
    Returns
    -------
    f : TYPE
        A figure object that shows the desired data with provided labels.

    """
    nrow = p_shape[0]
    ncol = p_shape[1]
    selectors = []
    for row in range(nrow):
        for col in range(ncol):
           selectors.append((row,col)) 
    f,ax_arr = plt.subplots(nrow,ncol)
    # X limits
    x_min = p_xlims[0]
    if x_min == p_xlims[1]: #if they're the same, get the whole range.
        x_max = p_data_dict[p_identifiers[0]][p_x_ref][-1]
    else:
        x_max = p_xlims[1]
        
    index = 0 #tracks which axis is being plotted.
    if len(p_data_dict[p_identifiers[0]][p_data_ref].shape) ==2:
        # Y limits for 2D
        y_min = p_ylims[0]
        if y_min == p_ylims[1]: #if they're the same, get the whole range.
            y_max = p_data_dict[p_identifiers[0]][p_y_ref][-1]
        else:
            y_max = p_ylims[1]

        for index in range(len(p_identifiers)):
            run = p_identifiers[index]
            label = run
            _,im = plot_and_return_2D_axis(
                ax_arr[selectors[index]],
                p_data_dict[run][p_data_ref],
                p_data_dict[run][p_x_ref],
                p_data_dict[run][p_y_ref],
                p_x_min = x_min,
                p_x_max = x_max,
                p_y_min = y_min,
                p_y_max = y_max,
                p_label = label,
                p_linear=p_linear)
            index+= 1
        cbar = f.colorbar(im, ax=ax_arr.ravel().tolist())
        cbar.ax.set_ylabel(p_units)
    
    if len(p_data_dict[p_identifiers[0]][p_data_ref].shape)==1:        
        for index in range(len(p_identifiers)):
            run = p_identifiers[index]
            label = run
            _,im = plot_and_return_1D_axis(
                ax_arr[selectors[index]],
                p_data_dict[run][p_data_ref]  ,  
                p_data_dict[run][p_x_ref],
                p_x_min = x_min,
                p_x_max = x_max,
                p_label = label,
                p_linear=True)    
            index+= 1
    # f.supylabel('Spectrogram time (s)')     #FUTURE RELEASE OF MATPLOTLIB
    # f.supxlabel('Spectrogram frequency (Hz)') # FUTURE RELEASE OF MATPLOTLIB
    f.text(0.5, 0.04, p_x_ref, ha='center')
    f.text(0.04, 0.5, p_y_ref, va='center', rotation='vertical')
    f.suptitle(p_data_ref + '\n' + p_subheading)
    return f

