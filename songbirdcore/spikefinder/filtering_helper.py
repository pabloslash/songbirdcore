
# Imports
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, filtfilt, freqz
import numpy as np


'''
FILTER FUNCTIONS
'''


# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# Build Butterworth filters with scipy.
def butter_filt_coefficients(fs, lowcut=[], highcut=[], btype='band', order=5):
    assert btype in ['band', 'low', 'high'], "Filter type must be 'low', 'high', or 'band'"
    if btype == 'low': assert lowcut, "Low cut frequency must be specified to build lowpass filter"
    elif btype == 'high': assert highcut, "High cut frequency must be specified to build a high filter"
    elif btype == 'low': assert lowcut, "Low and High cut frequencies must be specified to build a band filter"

    nyq = 0.5 * fs
    if lowcut: low = lowcut / nyq
    if highcut: high = highcut / nyq

    a, b = [], []
    if btype == 'band':
        b, a = butter(order, [low, high], btype)
    elif btype == 'low':
        b, a = butter(order, low, btype)
    elif btype == 'high':
        b, a = butter(order, high, btype)
    return b, a


# Load Butterworth filter coefficients from a file
def load_filter_coefficients_matlab(filter_file_path):
    coefficients = loadmat(filter_file_path)
    a = coefficients['a'][0]
    b = coefficients['b'][0]
    return b, a  # The output is a double list after loading .mat file


# Filter non-causally (forward & backwards) given filter coefficients
def noncausal_filter_1d(signal, b, a=1):
    y = filtfilt(b, a, signal)
    return y


def noncausal_filter_2d(data, b, a=1):
    """"
     Filter all channels in a 2D array [ Channels x Samples ]
     Noncausal filtering (filtfilt)

     Input: [ Channels x Samples ]
     b, a = Filter Coefficients

     Output: [ Filtered Channels x Samples ]
    """
    for ch in range(len(data)):
        data[ch] = noncausal_filter_1d(data[ch], b, a=a)
    return data


def noncausal_filter_3d(data, b, a=1):
    """"
     Filter all channels in a 3D array [ Epochs x Channels x Samples ]
     Noncausal filtering (filtfilt)

     Input: [ Epochs x Channels x Samples ]
     b, a = Filter Coefficients

     Output: [ Epochs x Filtered Channels x Samples ]
    """
    for ep in range(len(data)):
        data[ep] = noncausal_filter_2d(data[ep], b, a=a)
    return data


'''
DISCARD NOISY / SILENT CHANNELS
'''


def discard_channels_2d(neural_data, bad_channels, verbose=True):
    """"
     Deletes bad channels from a 2D array [channels x samples].

     Input: [Channels x Samples]
     bad_channels = bad channels to discard

     Output: [ (Channels - Bad_Channels) x Samples]
    """

    if verbose: print('Deleting channels {}'.format(bad_channels))
    return np.delete(neural_data, bad_channels, 0)


def discard_channels_3d(neural_data, bad_channels):
    """"
     Function for 'epoched data'.
     Deletes bad channels from a 3D array [epochs x channels x samples].

     Input: [Epochs x Channels x Samples]
     bad_channels = bad channels to discard

     Output: [ Epochs x (Channels - Bad_Channels) x Samples]
    """
    
    clean_neural_data = []
    for ep in range(len(neural_data)):
        clean_neural_data.append(discard_channels_2d(neural_data[ep], bad_channels))
        
    return clean_neural_data




