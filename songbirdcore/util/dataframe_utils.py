import pandas as pd
import numpy as np
import pickle
import songbirdcore.spikefinder.spike_analysis_helper as sh


"""Class to work with Pandas Dataframes of spike sorted and audio data"""

class Kilosort_Df:
    def __init__(self, spike_df_path, cluster_df_path, motif_dict_path: None, microphone_path: None):
        
        """Data file paths"""
        self.spike_df_path = spike_df_path
        self.cluster_df_path = cluster_df_path
        self.motif_dict_path = motif_dict_path
        self.microphone_path = microphone_path
        
        """Load dataframes"""
        # pd.core.frame.DataFrame
        with open(self.spike_df_path, 'rb') as f:
            self.spk_df = pickle.load(f)
        
        # pd.core.frame.DataFrame
        with open(self.cluster_df_path, 'rb') as f:
            self.clu_df = pickle.load(f)
    
        # pd.core.frame.DataFrame
        with open(self.motif_dict_path, 'rb') as f:
            self.mot_dict = pickle.load(f)
        
        # Numpy.array
        self.audio = np.load(self.microphone_path)
        
        """Variables"""
        self.fs_ap = self.mot_dict['s_f_ap_0']
        self.fs_audio = self.mot_dict['s_f']

    
    def get_window_spikes(self, clu_list: np.array, start_sample: int, end_sample: int, clean_raster=False) -> np.array:
        """
        Return array of spiking events for clusters of interest (clu_list) within a window of interest (start_sample -> end_sample)

        Params:
            spk_df: dataframe containing ALL spikes specifying cluster, nucleus, spike-time, main channel and ksort label.
            clu_list: list of clusters to include in the raster plot
            start_sample: start sample of the window of interest
            end_sample: end sample of the window of interest
        Keyword Params:
            clean_raster: bool. Remove events that co-occur across channels

        Return: np.array of spiking events within the window of interest [m_clusters x n_timestamps]
        """

        # Get all spikes that occur within the time window of interest (start_sample -> end_sample)
        spk_t = self.spk_df.loc[self.spk_df['times'].between(start_sample, end_sample, "left")] # The right-most sample corresponds to the array edge

        # Build array of spiking events [m_clusters x n_timestamps]
        spk_arr = np.zeros((clu_list.size, end_sample - start_sample))
        for i, clu_id in enumerate(clu_list):
            # For each cluster, get all the times when a spike occurs and populate array
            clu_spk_t = spk_t.loc[spk_t['clusters']==clu_id, 'times'].values
            spk_arr[i, clu_spk_t - start_sample] = 1

        if clean_raster:
            spk_arr = sh.clean_spikeRaster_noisyEvents2d(spk_arr) # Remove noisy (simultaneous) events
            spk_arr = sh.remove_silent_channels_2d(spk_arr)

        return spk_arr
    
        
    def get_rasters_spikes(self, clu_list: np.array, start_samp_list: np.array, span_samples: int) -> np.array:
        """
        Return raster of spiking events for clusters of interest (clu_list) within a LIST OF WINDOWS of interest of the same length (start_sample -> start_sample+span_samples)

        Params:
            spk_df: dataframe containing ALL spikes specifying cluster, nucleus, spike-time, main channel and ksort label.
            clu_list: list of clusters to include in the raster plot
            start_samp_list: list of start samples of interest
            span_samples: length of the time window (number of samples)

        Return: np.array of spiking events within the window of interest [m_clusters x n_timestamps] for each period of interest
        """

        # Build array of spiking events [m_clusters x n_timestamps] for the clusters of interes, for each period specified by the start_samp_list
        spk_arr_list = [self.get_window_spikes(clu_list, i, i+span_samples) for i in start_samp_list]
        return np.stack(spk_arr_list, axis=-1)


    def get_rasters_audio(self, start_sample_list: list, span_samples: int) -> np.array:
        """
        Return array of snippets of audio specified by start_sample_list (start_sample -> start_sample+span_samples)

        Params:
            audio: audio signal
            start_sample_list: list of start_times to include in the audio array
            span_samples: length of the time window of interest (number of samples)

        Return: np.array of snippets of audio [m_audio_snippets x n_timestamps] for each period of interest
        """
        audio_arr_list = [self.audio[start_sample_list[i] : start_sample_list[i]+span_samples] for i in range(len(start_sample_list))]
        return(np.array(audio_arr_list).squeeze().tolist())
    
    
    def print_neural_clusters(self):
        """
        Print number of each cluster type found in each nucleus of interest.
        """
        nuclei = ['hvc', 'ra']
        qualities = ['SUA1', 'SUA2', 'SUA3', 'MUA']

        for n in nuclei:
            for q in qualities:
                print(n+' '+q+':', len(np.unique(self.clu_df.loc[(self.clu_df['quality']==q) & (self.clu_df['nucleus'].isin([n])), 'cluster_id'])))
                
    
    def get_isi(self, cluster_id:int) -> np.array:
        """"
        Calculate Inter-Spike-Intervals for one neural cluster in spk_df.

        Arguments:
            spk_df = dataframe of events [times, cluster_id, nucleus, main_chan etc.]
            cluster_id = cluster number in dataframe
        Return:
            Numpy array of inter-spike-interval times
        """
        clu_sel = self.spk_df['cluster_id']==cluster_id
        isi = np.diff(self.spk_df.loc[clu_sel].times)
        return isi
    
    def save_clu_df_as_pickle(self, save_path: str):
        """"
        Save curated cluster dataframe to location 'save_path'.
        Warning: If path = self.cluster_df_path, the file will be OVERWRITTEN.
        """
        self.clu_df.to_pickle(save_path)