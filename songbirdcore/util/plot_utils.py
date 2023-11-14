import seaborn as sns
from matplotlib import pyplot as plt
import copy
import numpy as np
from numba import prange
import random
from matplotlib.gridspec import GridSpec
import songbirdcore.util.label_utils as luts
from songbirdcore.style_params import syl_colors


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


def plot_2D_trajectories(trajectories: np.array, trajectory_labels: np.array=None, dimensions2plot: list=[0, 1], num_trials2plot=10):
    
    """
        trajectories: np.array [trials x dimensions x time]
        trajectory_labels: np.array [trials x time]
    """
    
    colors = syl_colors
    
    if trajectory_labels is not None:
        assert (trajectories.shape[0]==trajectory_labels.shape[0]) & (trajectories.shape[2] == trajectory_labels.shape[1]), "Trajectory and label dimensions mismatch."
    
    # 2D plot
    fig, ax = plt.subplots(nrows=1, figsize=(7, 7))#, constrained_layout=True)
    
    if trajectory_labels is not None:
        lbl_edges = luts.TextgridLabels.find_label_edges_2D(trajectory_labels)

        for t in range(min(len(trajectories), num_trials2plot)):
            lbl_t = lbl_edges[t]
            for idx in range(len(lbl_t)-1):
                plt_color = colors[ trajectory_labels[t, lbl_t[idx]] ]      # Choose color based on label for this segment

                start = lbl_t[idx]                                  # Plot starting at the end of the previous segment (except if it is the first segment)
                end   = lbl_t[idx+1] + 1 if idx!=len(lbl_t)-2 else lbl_t[idx+1]      # Plot samples until the start of the next segment
                x_linspace = list(range(start, end))       
                ax.plot(trajectories[t][dimensions2plot[0]][start:end], trajectories[t][dimensions2plot[1]][start:end], linewidth=2, color=plt_color)
    else:
        for t in range(min(len(trajectories), num_trials2plot)):
            ax.plot(trajectories[t][dimensions2plot[0]], trajectories[t][dimensions2plot[1]], linewidth=2)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.xaxis.set_tick_params(labelsize=25)

    return fig
    
    
def plot_3D_trajectories(trajectories: np.array, trajectory_labels: np.array=None, dimensions2plot: list=[0, 1, 2], num_trials2plot=10):

    """
        trajectories: np.array [trials x dimensions x time]
        trajectory_labels: np.array [trials x time]
    """
    
    colors = syl_colors
    
    if trajectory_labels is not None:
        assert (trajectories.shape[0]==trajectory_labels.shape[0]) & (trajectories.shape[2] == trajectory_labels.shape[1]), "Trajectory and label dimensions mismatch."
        
    # 3D plot
    fig = plt.figure(constrained_layout=True, figsize=(30, 12))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2, 1])
    gs.update(wspace=0, hspace=0)
    ax = fig.add_subplot(gs[:,0], projection='3d')

    if trajectory_labels is not None:
        lbl_edges = luts.TextgridLabels.find_label_edges_2D(trajectory_labels)

        for t in range(min(len(trajectories), num_trials2plot)):
            lbl_t = lbl_edges[t]

            for idx in range(len(lbl_t)-1):
                plt_color = colors[ trajectory_labels[t, lbl_t[idx]] ]      # Choose color based on label for this segment
                start = lbl_t[idx]               # Plot starting at the end of the previous segment (except if it is the first segment)
                end   = lbl_t[idx+1]+1 if idx !=  len(lbl_t)-2 else lbl_t[idx+1]      # Plot samples until the start of the next segment              
                ax.plot3D(trajectories[t][dimensions2plot[0]][start:end], trajectories[t][dimensions2plot[1]][start:end], trajectories[t][dimensions2plot[2]][start:end],
                         linewidth=2.5, color=plt_color)
    else:
        for t in range(min(len(trajectories), num_trials2plot)):
            ax.plot(trajectories[t][dimensions2plot[0]], trajectories[t][dimensions2plot[1]], trajectories[t][dimensions2plot[2]], linewidth=2)

    ax.set_xlim(-2, 3.5)
    ax.set_zticks([-2, -1, 0, 1])
    ax.xaxis.set_tick_params(labelsize=25)
    ax.yaxis.set_tick_params(labelsize=25)
    ax.zaxis.set_tick_params(labelsize=25)
    ax.view_init(225, 225)
    ax.xaxis.set_pane_color((1, 1, 1, 0.1))
    ax.yaxis.set_pane_color((1, 1, 1, 0.1))
    ax.zaxis.set_pane_color((1, 1, 1, 0.1))

    # 1D plot
    for d in dimensions2plot:
        ax = fig.add_subplot(gs[d, 1])
        for t in range(min(len(trajectories), num_trials2plot)):
            
            if trajectory_labels is not None:
                lbl_t = lbl_edges[t]
                for idx in range(len(lbl_t) -1 ):

                    plt_color = colors[ trajectory_labels[t, lbl_t[idx]] ]      # Choose color based on label for this segment
                    start = lbl_t[idx]                                  # Plot starting at the end of the previous segment (except if it is the first segment)
                    end   = lbl_t[idx+1] + 1 if idx !=  len(lbl_t)-2 else lbl_t[idx+1]      # Plot samples until the start of the next segment
                    x_linspace = list(range(start, end))       
                    ax.plot(x_linspace, trajectories[t][d][start:end], linewidth=2.5, color=plt_color)
            else:
                ax.plot(trajectories[t][d], linewidth=2)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin-0.1, ymax+0.1)
        ax.get_xaxis().set_visible(False)
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.yaxis.set_tick_params(labelsize=25)
    
    return fig