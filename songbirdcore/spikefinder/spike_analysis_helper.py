
# Imports
import numpy as np
import copy
from scipy import stats

'''
THRESHOLD CROSSINGS
'''


def calculate_signal_rms(signal):
    """"
     Returns the root mean square {sqrt(mean(samples.^2))} of a 1D vector.
    """
    return np.sqrt(np.mean(np.square(signal)))



def find_threshold_crossings_1d(neural_channel, fs, th=3.5, ap_time=1):
    """"
     Converts a 1D-list (1 channel) of raw extracellurar continuous neural recordings into a binary matrix indicating
       where threshold crossings occur.

     Input: neural_channel[1 x Samples]
     fs = Sampling frequency of input data
     th = Number of standard deviations below RMS of each channel considered to trigger an AP
     ap_time (ms) = 'Artificial' refractory period. Time between 2 found AP to consider them independent.
       Typycal depolarization time is ~1ms.

     Output: [1 x Samples]
    """
    samples = len(neural_channel)
    th_crossings = np.zeros(samples)

    # Find RMS and threshold idxs
    rms = calculate_signal_rms(neural_channel)
    th_idx = [idx for idx, val in enumerate(neural_channel) if val < -th*rms]

    # Delete idx if they belong to the same AP
    ap_samples = int(ap_time / 1000 * fs)

    clean_th_idx = []
    last_idx = -ap_samples  # For first AP

    for i in th_idx:
        if i > last_idx + ap_samples:
            clean_th_idx.append(i)
            last_idx = i

    th_crossings[clean_th_idx] = 1

    return th_crossings


def find_threshold_crossings_2d(neural_data, fs, th=3.5, ap_time=1, verbose=False):
    """"
     Converts a 2D-list (n-channels) of raw extracellurar continuous neural recordings into a binary matrix indicating
       where threshold crossings occur.

     Input: neural_data[Channels x Samples]
     fs = Sampling frequency of input data
     th = Number of standard deviations below RMS of each channel considered to trigger an AP
     ap_time (ms) = 'Artificial' refractory period. Time between 2 found AP to consider them independent.
       Typycal depolarization time is ~1ms.

     Output: [Channels x Samples]
    """
    channels = len(neural_data)
    samples = len(neural_data[0])
    spike_raster = np.zeros([channels, samples])

    # Populate spike raster matrix
    for ch in range(channels):
        if verbose: print('Finding threshold crossings in channel {}'.format(ch))
        spike_raster[ch] = find_threshold_crossings_1d(neural_data[ch], fs, th=th, ap_time=ap_time)

    return spike_raster


# Return binned vector according to bin size (Input: 1D list)
def downsample_list_1d(dat, number_bin_samples, mode='sum'):
    """"
     Downsamples (bins) a 1D-list acording to selected number of bin samples.

     Input: dat[1 x Samples]
     number_bin_samples = Number of samples in each bin (bin size in samples).
     mode: 'sum', 'mean', 'mode' (downsample summing, averaging or taking the mode of all samples)

     Output: [1 x Samples]
    """
    assert mode in ['sum', 'mean', 'mode'], "Mode type must be 'sum', 'mean', or 'mode', default is 'sum'"
    
    if mode == 'sum':
        return np.nansum(np.array(dat[:(len(dat) // number_bin_samples) * number_bin_samples]).reshape(-1, number_bin_samples), axis=1)
    
    elif mode == 'mean':
        return np.nanmean(np.array(dat[:(len(dat) // number_bin_samples) * number_bin_samples]).reshape(-1, number_bin_samples), axis=1)
    
    elif mode == 'mode':
        return stats.mode(np.array(dat[:(len(dat) // number_bin_samples) * number_bin_samples]).reshape(-1, number_bin_samples), axis=1)[0].reshape(-1)

    
# Return binned matrix along dimension 2 according to bin size (Input: 2D list)
def downsample_list_2d(dat, number_bin_samples, mode='sum'):
    """"
     Downsamples (bins) a 2D-list acording to selected number of bin samples.

     Input: dat[n x Samples]
     number_bin_samples = Number of samples in each bin (bin size in samples).
     mode: 'sum', 'mean', 'mode' (downsample summing, averaging or taking the mode of all samples)

     Output: [n x Samples]
    """
    downsampled_dat = []
    for i in range(np.array(dat).shape[0]):
        downsampled_dat.append(downsample_list_1d(dat[i], number_bin_samples, mode=mode))

    return np.array(downsampled_dat)


# Return binned matrix along dimension 3 according to bin size (Input: 3D list)
def downsample_list_3d(dat, number_bin_samples, mode='sum'):
    """"
     Downsamples (bins) a 3D-list acording to selected number of bin samples.

     Input: dat[trials x n x Samples]
     number_bin_samples = Number of samples in each bin (bin size in samples).
     mode: 'sum', 'mean', 'mode' (downsample summing, averaging or taking the mode of all samples)

     Output: [trials x n x Samples]
    """
    downsampled_dat = []
    for trial in range(np.array(dat).shape[0]):
        downsampled_dat.append(downsample_list_2d(dat[trial], number_bin_samples, mode=mode)) 
        
    return np.array(downsampled_dat)


def detect_noisy_silent_channels_3d(data, rms_deviation_threshold=3, verbose=True):
    """"
     Suggests channels to be discarded from analysis because they are too noisy or too silent compared to the rest.
     Any channel with a RMS deviated above / below 'RMS_deviation_threshold' number of SDs of the RMS distribution accross all channels is identified.
     Deletion of these channels is suggested.

     Input:  3D List of 2D Numpy Arrays: [Epochs x Channels x Raw Samples]
     rms_deviation_threshold = Number of standard deviations above / below the mean of the signal accross all chanels to consider a channel 'noisy' or 'silent'.
     
     Output: List of suggested channels to discard
    """
    
    # Concatenate all epochs.
    num_channels = len(data[0])
    concat_data = np.empty([num_channels, 0])
    
    for ep in range(len(data)):
        concat_data = np.append(concat_data, data[ep], axis=1)
        
    # Detect noisy channels as if data were continuous [channels x samples]
    return detect_noisy_silent_channels_2d(concat_data, rms_deviation_threshold=rms_deviation_threshold, verbose=verbose)


def detect_noisy_silent_channels_2d(data, rms_deviation_threshold=3, verbose=True):
    """"
     Suggests channels to be discarded from analysis because they are too noisy or too silent compared to the rest.
     Any channel with a RMS deviated above / below 'RMS_deviation_threshold' number of SDs of the RMS distribution accross all channels is identified.
     Deletion of these channels is suggested.

     Input:  2D Numpy Array: [Channels x Raw Samples]
     Output: List of suggested channels to discard
    """
    
    # In case std was computed from 3D neural data [epochs x ch x samples]
    rms_channels = [calculate_signal_rms(ch) for ch in data] 
    rms_distribution_mean = np.mean(rms_channels)
    rms_distribution_sdt = np.std(rms_channels)
    
    broken_channels = [idx for idx, val in enumerate(rms_channels) if (val >= rms_distribution_mean + rms_deviation_threshold*rms_distribution_sdt or val <= rms_distribution_mean - rms_deviation_threshold*rms_distribution_sdt)]
    if verbose: print('Suggested broken channels: {}'.format(broken_channels))
        
    return broken_channels


def remove_silent_channels_2d(data, verbose=True):
    """"
     Remove silent channels

     Input:  2D Numpy Array: [Channels x Raw Samples]
     Output: 2D Numpy Array: [Channels x Raw Samples]
    """
    # Remove rows having all zeroes
    if verbose: print('Removing {} silent channels'.format(len(data[np.all(data == 0, axis=1)])))
    return data[~np.all(data == 0, axis=1)]

    
def clean_spikeRaster_noisy_events_3d(spike_raster, coocurrence_threshold=10, verbose=True):
    """"
     TIP: Feed data binned to 1ms bins.
     Deletes detected spikes from a spike raster. [Ch x Binary samples (0/1)] 
     Any bin with combined spike counts of more than 'coocurrence_threshold' SD of the distribution of spike counts accross the entire dataset 
     is identified. Spikes in that bin are erased from all channels.

     Input:  3D list of 2D Numpy Arrays [Epochs x Channels x Samples]
     coocurrence_threshold = Number of standard deviations (of spike sum accross channels) above the mean to be considered a noisy event.
                             This value may depend on the typicial activity of the brain region of interest, the number of channels etc.
                             
     Output: 3D list of 2D Numpy Arrays [Epochs x Channels x Samples]
    """
    
    # Concatenate all epochs and calculate the standard deviation of the spike distribution accross all channels along time.
    concat_spike_raster = np.array([])
    for ep in range(len(spike_raster)):
        concat_spike_raster = np.append(concat_spike_raster, np.sum(spike_raster[ep], 0), axis=0)
    
    spike_mean = np.mean(concat_spike_raster)
    spike_sdt = np.std(concat_spike_raster)
    
    # Clean time events in which the coocurrence of spikes is 'coocurrence_threshold' times the standard deviation of the distribution
    cl_spike_raster = []
    for ep in range(len(spike_raster)):
        cl_spike_raster.append(clean_spikeRaster_noisyEvents2d(spike_raster[ep], spike_distribution=[spike_mean, spike_sdt], coocurrence_threshold=coocurrence_threshold))
        
    return cl_spike_raster


def clean_spikeRaster_noisyEvents2d(spike_raster, spike_distribution=None, coocurrence_threshold=10, verbose=True):
    """"
     Deletes detected spikes from a spike raster.
     Any bin with combined spike counts of more than 'coocurrence_threshold' SD of the distribution of spike counts accross the entire recording 
     is identified. Spikes in that bin are deleted in all channels.

     Input:  2D Numpy Array: [Channels x Samples]
     spike_distribution    = Mean and standard deviation of the spike distribution (sum of spikes in each time step). 
                             Typically passed as an argument when the statistics are calculated from 3D data [Epochs x Channels x Samples].
     coocurrence_threshold = Number of standard deviations (of spike sum accross channels) above the mean to be considered a noisy event.
                             This value may depend on the typicial activity of the brain region of interest, the number of channels etc.
     
     Output: 2D Numpy Array: [Channels x Samples] where spikes have been erased in noisy events.
    """
    
    # In case std was computed from 3D neural data [epochs x ch x samples]
    if spike_distribution == None:
        spike_mean = np.mean(np.sum(spike_raster, 0))
        spike_sdt = np.std(np.sum(spike_raster, 0))
    else:
        spike_mean = spike_distribution[0]
        spike_sdt = spike_distribution[1]
    
    # Clean time events in which the coocurrence of spikes is 'coocurrence_threshold' times the standard deviation of the distribution
    cl_spike_raster = copy.deepcopy(spike_raster)
    
    noisy_idx = [idx for idx, val in enumerate(np.sum(spike_raster, 0)) if val >= spike_mean + coocurrence_threshold*spike_sdt]
        
    if verbose: print('Cleaning {} noisy events'.format(len(noisy_idx)))
    cl_spike_raster[:, noisy_idx] = 0
        
    return cl_spike_raster


def CAR_2d(neural_data, verbose=True):
    """"
     Performs Common Average Reference on neural data structured as [Channels x Samples].
     Input:  2D Numpy Array: [Channels x Samples]
     Output: 2D Numpy Array: [Channels x Samples] where with CAR perfomed.
    """
    CA = np.mean(neural_data, axis=0)
    return neural_data - CA
