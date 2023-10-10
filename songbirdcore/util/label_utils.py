import numpy as np
from praatio import tgio


class TextgridLabels:
    
    """ This class can be used to work with a .Textgrid file containing annotated labels for an audio file, e.g. using Praat"""
        
    def __init__(self, textgrid_file_path: str, audio_file_path: str, fs_audio: float, labels_tier: str='syllables') -> list:

        self.textgrid_file_path = textgrid_file_path
        self.audio_file_path = audio_file_path
        self.fs_audio = fs_audio
        
        """Load data"""
        self.audio = np.load(self.audio_file_path)   # Numpy.array 
        self.labels = self.load_labels_from_textgrid(labels_tier=labels_tier)

        
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
        entryList = tg.tierDict[labels_tier].entryList # Get all intervals

        # Create list of labels
        labels = []
        for i, interval in enumerate(entryList):
            # ! Round the fs_audio & the interval times (for number of samples in labels and audio array to match)
            labels.extend([interval.label for n in range(round(interval.end*round(self.fs_audio)) - round(interval.start*round(self.fs_audio)))])
        return labels


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
        """
        Find all indexes in a list of labels [1 x labels] where a change of label occurs.
        fnx(np.where) points to the index of the last number before a change occurs, so we add 1 to the index vector.
        Add index 0 and last index (len(mot)) for plotting purposes.
        """
        lbl_edges = np.where(np.diff(labels))[0] + 1 
        lbl_edges = np.insert(lbl_edges, 0, 0)
        lbl_edges = np.append(lbl_edges, len(labels))
        return lbl_edges


    @staticmethod
    def find_label_edges_2D(labels_array: list) -> list:
        """
        Find all indexes in array of labels [epochs x labels] where a change of label occurs.
        """
        lbl_edges = []
        for e in range(len(labels_array)):
            lbl_edges.append(TextgridLabels.find_label_edges(labels_array[e]))
        return lbl_edges