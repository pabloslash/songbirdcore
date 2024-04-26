import numpy as np
from praatio import tgio
from scipy.interpolate import make_interp_spline
import warnings

class TextgridLabels:
    
    """ This class can be used to work with a .Textgrid file containing annotated labels for an audio file, e.g. using Praat"""
        
    def __init__(self, textgrid_file_path: str, audio_file_path: str, fs_audio: float, labels_tier: str='labels') -> list:

        self.textgrid_file_path = textgrid_file_path
        self.audio_file_path = audio_file_path
        self.fs_audio = fs_audio
        
        """Check TextGrid"""
        self.missing_label_intervals = self.find_missing_label_intervals(labels_tier=labels_tier)
        
        """Load data"""
        self.audio = np.load(self.audio_file_path)   # Numpy.array 
        self.labels = self.load_labels_from_textgrid(labels_tier=labels_tier)
        
        if len(self.audio)!=len(self.labels): 
            warnings.warn("WARNING: audio length is different than labels length.")
            print('Len audio: ', len(self.audio), ', len labels: ', len(self.labels))

        
    def load_labels_from_textgrid(self, labels_tier: str='syllables') -> list:
        """
        Load Textgrid file containing human annotations of an audio recording.

        Arguments:
            textgrid_path: Path to Textgrid file.
            fs_audio: sampling frequency of the original audio file.

        Keyword Arguments:
            labels_tier: tier within the Textgrid file containing labels of interest.
                !TODO: Modify this function to read multiple tiers if desired.
        """
        tg = tgio.openTextgrid(self.textgrid_file_path)
        print('Tier Name List: ', tg.tierNameList)
        entryList = tg.tierDict[labels_tier].entryList # Get all intervals

        # Create list of labels
        labels = []
        for i, interval in enumerate(entryList):
            # ! Round the fs_audio & the interval times (for number of samples in labels and audio array to match)
            labels.extend([interval.label for n in range(round((interval.end - interval.start)*round(self.fs_audio)))])
        return labels

                                                            
    def find_missing_label_intervals(self, labels_tier: str='labels'):
        """
            Find intervals (in seconds) od missing labels.
            Return a list of time intervals [start_s, end_s] where the labels.Textgrid file is missing a label.
        """
        tg = tgio.openTextgrid(self.textgrid_file_path)
        entryList = tg.tierDict[labels_tier].entryList # Get all intervals                                                  
                    
        # Create list of 'missing-label times' in seconds
        missing_label_intervals = []
        for i in range(len(entryList)-1):
            if (entryList[i].end != entryList[i+1].start): 
                print('Missing label in interval: ', i, entryList[i].end, entryList[i+1].start)
                missing_label_intervals.append([entryList[i].end, entryList[i+1].start])

        if not missing_label_intervals: print('All intervals in the tier of interest of the .Textgrid file have been labeled (no missing labels).')
        else: print('The following intervals in the .TextGrid file have not been labeled: {}'.format(missing_label_intervals))
        
        return missing_label_intervals                                       
                                                            
                                                            
    def find_label_starts(self, target_label: int) -> np.array:
        """
            Return an array of indexes where the target label occurs starts in a list of labels.
            This fxn may be used to find the start of all occurrences of a given syllable/motif.

            Arguments:
                target_label: label of interest for which to find first occurrences
            Returns:
                Numpy array containing the indexes where the target_label starts throughout the labels list.
        """
        return np.array([l for l in range(1, len(self.labels)) if (self.labels[l]==target_label and self.labels[l-1]!=target_label)])


    def find_bout_limits(self, start_bout_label: int=1, end_bout_label: int=8):
        """
            Find [start, end] indexes of all bouts starting with start_bout_label.

            Arguments:
                start_bout_label: expected label to occur at the start of a bout (typically syllable 1).
                out_of_bout_label: expected label to occur at the end of a bout (typically an out-of-bout silence)
            Returns:
                Numpy array containing the [start, end] indexes of each bout in the list.
        """

        bout_limits = []
        in_bout = False

        for l in range(0, len(self.labels)):

            # If the current index corresponds to the start_bout_label and the previous index is out_of_bout:
            if (self.labels[l]==start_bout_label and in_bout==False): 
                start_bout = l
                in_bout = True

            # If current index corresponds to an out_of_bout_label after a bout period:
            elif ((self.labels[l]==end_bout_label and in_bout==True) or (l==len(self.labels))): 
                end_bout = l
                in_bout = False
                bout_limits.append([start_bout, end_bout])

        return np.array(bout_limits)
    
    
    def get_rasters_labels(self, start_sample_list: list, span_samples: int) -> list:
        """
        Return array of snippets of labels specified by start_sample_list (start_sample -> start_sample+span_samples)

        Params:
            start_sample_list: list of start_times to include in the labels array
            span_samples: length of the time window of interest (number of samples)

        Return: np.array of snippets of labels [m_label_snippets x n_timestamps] for each period of interest
        """
        label_arr_list = [self.labels[start_sample_list[i] : start_sample_list[i]+span_samples] for i in range(len(start_sample_list))]
        return(np.array(label_arr_list).squeeze().tolist())
    

    @staticmethod
    def find_label_edges(labels: list) -> np.array:
        ''' 
        Find all indexes in a list of labels [1 x labels] where a change of label occurs.
        fnx(np.where) points to the index of the last number before a change occurs, so we add 1 to the index vector.
        Add index 0 and last index (len(mot)) for plotting purposes.
        
        Parameters:
            labels: tends to be a 1d list of integers where each element indicates the label of a single sample of a corresponding signal.
        Output:
            1D list containing the indexes where a label change occurs in labels.
        '''
        lbl_edges = np.where(np.diff(labels))[0] + 1 
        lbl_edges = np.insert(lbl_edges, 0, 0)
        lbl_edges = np.append(lbl_edges, len(labels))
        return lbl_edges


    @staticmethod
    def find_label_edges_2D(labels_array: list) -> list:
        ''' 
        Find indexes in array of labels [trials x label_signals] where a change of label occurs.
        
        Parameters:
            label_signal_2d: 2D list / np.ndarray of trials x label_signals. 
            A label signal tends to be a 1d list of integers where each element indicates the label of a single sample of a corresponding signal.
        Output:
            2D list of [motifs x change of label indexes] containing the indexes along all trials where a label change occurs.
        '''
        lbl_edges = []
        for e in range(len(labels_array)):
            lbl_edges.append(TextgridLabels.find_label_edges(labels_array[e]))
        return lbl_edges


    @staticmethod
    def list_expander(list_1d: list, expand_multiplier:int) -> np.ndarray:
        '''
        Extend a list or array by repeating each element in the array (n) times
        
        Parameters:
            list_1d: 1D list or array to be expanded
            expand_multiplier: (n) times to repeat each element in list_1d
        Output:
            1D array with length: len(list_1d) * expand_multiplier
        '''
        return np.array([[i]*expand_multiplier for i in list_1d]).reshape(expand_multiplier*len(list_1d))
    

    @staticmethod
    def signal_smoother(signal: np.ndarray, interp_multiplier: int, interp_order: int):
        '''
        Smooth an input signal using an interpolating B-spline
        
        Parameters:
            signal: 1D numpy.ndarray to be smothed
            interp_multiplier: defines the final number of samples of the smoothed signal
            interp_order: order of the spline to fit to the input signal
        Output:
            1D smoothed signal with length: len(signal) * interp_multiplier
        '''
        
        # Number of samples in original signal
        x_linspace = np.linspace(0, len(signal), len(signal))
        # Number of samples in smoothed signal
        x_linspace_smooth = np.linspace(0, len(signal), interp_multiplier*len(signal))    
        # Signal interpolator
        spline = make_interp_spline(x_linspace, signal, k=interp_order)
        return spline(x_linspace_smooth)

