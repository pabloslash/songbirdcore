import seaborn as sns
from matplotlib import pyplot as plt
import copy
import numpy as np
from numba import prange
import random


def plot_as_raster(x, ax=None, t_0=None, s=20, color=None):
    """"
    Plot a binary array of neural events as a raster. 
    
    Arguments:
        x = binary array of events [n_clusters, n_timestamps]
    Keyword Arguments:
        ax = figure axis
        t_0 = sample of interest. If a given sample is given, a green vertical line is drawn to mark it. Default: None
        s = Marker size
        color = Color of the event dots in the raster. Default: random
    """
    
    x_array = copy.deepcopy(x)
    n_y, n_t = x_array.shape
    
    row = np.ones(n_t) + 1
    t = np.arange(n_t)
    col = np.arange(n_y)
    
    frame = col[:, np.newaxis] + row[np.newaxis, :]
    x_array[x_array==0] = np.nan
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if color == None:
        facecolor=(random.random(),random.random(),random.random())
    else:
        facecolor=color
            
    raster = ax.scatter(t * x_array, frame * x_array, marker='.', facecolor=facecolor, s=s, rasterized=False)
    
    if t_0 is not None:
        ax.axvline(x=t_0, color='green')
        
    return ax


def plottable_array(x:np.ndarray, scale:np.ndarray, offset:np.ndarray) -> np.ndarray:
    """ Rescale and offset an array for quick plotting multiple channels, along the 
        1 axis, for each jth axis
    Arguments:
        x {np.ndarray} -- [n_col x n_row] array (each col is a chan, for instance)
        scale {np.ndarray} -- [n_col] vector of scales (typically the ptp values of each row)
        offset {np.ndarray} -- [n_col] vector offsets (typycally range (row))

    Returns:
        np.ndarray -- [n_row x n_col] scaled, offsetted array to plot
    """
    # for each row [i]:
    # - divide by scale_i
    # - add offset_i
    n_row, n_col = x.shape
    for col in prange(n_col):
        col_mean = np.mean(x[:, col])
        for row in range(n_row):
            x[row, col] = (x[row, col] - col_mean)* scale[col] + offset[col]
    return x


def plot_array(x: np.ndarray, scale='each', ax=None, offset_scale=1, linewidth=1, color=None):

    """ Rescale and offset an array for quick plotting multiple channels, along the 
        1 axis, for each jth axis
    Arguments:
        x {np.ndarray} -- [n_col x n_row] array (each col is a chan, for instance)
    
    Keyword Arguments:
        scale {str} -- {'each', 'max'} (default: {'each'}) whether to scale within each col
                        or to the max ptp of all cols
        ax {[type]} -- [description] (default: {None})
    """
    if ax is None:
        _, ax = plt.subplots()
    
    # arrange the array:
    n_row, n_col = x.shape
    offset = np.arange(n_col) * offset_scale
    ptp = np.ptp(x, axis=0)
    if scale == 'max':
        ptp[:] = np.max(ptp)
    
    x_scaled = plottable_array(x, 1./ptp, offset)
    ax.plot(x_scaled, color=color, linewidth=linewidth)