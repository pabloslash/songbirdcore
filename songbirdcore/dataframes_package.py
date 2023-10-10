import pandas as pd
import numpy as np
import pickle
import songbirdcore.spikefinder.spike_analysis_helper as sh


class AudioDictionary:
    
    """Class to work with a Dictionary of relevant params and timestamps in neural and audio data"""
    
    def __init__(self, audio_array: np.array, all_syn_dict: dict) -> dict:

        """
        Arguments:
            audio_array: Array [n_audio_segments, 2] with [start, end] of the segments of interest in each row, in seconds
            all_syn_dict: Dictionary of synced streams of interest (e.g. ap_0, lf_0, nidq, wav) 
        """
        
        s_f_wav = all_syn_dict['wav']['s_f']

        start_ms = (audio_array[:, 0]*1000).astype(np.int64)
        len_ms = (np.diff(audio_array)*1000).astype(np.int64).flatten()

        self.audio_dict = {
                's_f': s_f_wav, # s_f used to get the spectrogram
                's_f_nidq': all_syn_dict['nidq']['s_f'],
                's_f_ap_0': all_syn_dict['ap_0']['s_f'],
                'start_ms': start_ms,
                'len_ms': len_ms,
                'start_sample_naive': (start_ms * s_f_wav * 0.001).astype(np.int64),
                'start_sample_nidq': np.array([np.where(all_syn_dict['nidq']['t_0'] > start)[0][0] for start in start_ms*0.001]),
                'start_sample_wav': np.array([np.where(all_syn_dict['wav']['t_0'] > start)[0][0] for start in start_ms*0.001])
               }

        # start_ms_ap_0 = all_syn_dict['wav']['t_p'][self.audio_dict['start_sample_wav']]*1000
        start_ms_ap_0 = all_syn_dict['nidq']['t_p'][self.audio_dict['start_sample_nidq']]*1000

        self.audio_dict['start_ms_ap_0'] = start_ms_ap_0
        self.audio_dict['start_sample_ap_0'] = np.array([np.where(all_syn_dict['ap_0']['t_0'] > start)[0][0] for start in start_ms_ap_0*0.001])
        self.audio_dict['start_sample_ap_0'] = (self.audio_dict['start_sample_ap_0']).astype(np.int64)


    def save_audio_dict_as_pickle(self, save_path: str):
        """"
        Save audio_dictionary to location 'save_path'.
        """
        
        with open(save_path, 'wb') as h:
            pickle.dump(self.audio_dict, h, protocol=pickle.HIGHEST_PROTOCOL)
        
        
class SortDataframe:
    
    """Class to work with Pandas Dataframes of spike sorted (kilosort) and audio data"""
    
    def __init__(self, spike_df_path, cluster_df_path, audio_dict_path: None, audio_file_path: None):
        
        """Load data"""
        self.spike_df_path = spike_df_path
        self.cluster_df_path = cluster_df_path
        
        with open(self.spike_df_path, 'rb') as f:
            self.spk_df = pickle.load(f) # pd.core.frame.DataFrame
        
        with open(self.cluster_df_path, 'rb') as f:
            self.clu_df = pickle.load(f) # pd.core.frame.DataFrame
        
        if audio_dict_path:
            
            self.audio_dict_path = audio_dict_path
            with open(self.audio_dict_path, 'rb') as f:
                self.audio_dict = pickle.load(f) # pd.core.frame.DataFrame

            """Sampling frequencies"""
            self.fs_ap = self.audio_dict['s_f_ap_0']
            self.fs_audio = self.audio_dict['s_f']
            
        if audio_file_path:
            
            self.audio_file_path = audio_file_path
            self.audio = np.load(self.audio_file_path) # Numpy.array

    
    def get_window_spikes(self, clu_list: np.array, start_sample: int, end_sample: int, clean_raster=False, verbose=True) -> np.array:
        
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
        spk_t = self.spk_df.loc[self.spk_df['times'].between(start_sample, end_sample, "left")]

        # Build array of spiking events [m_clusters x n_timestamps]
        spk_arr = np.zeros((clu_list.size, end_sample - start_sample))
        for i, clu_id in enumerate(clu_list):
            # For each cluster, get all the times when a spike occurs and populate array
            clu_spk_t = spk_t.loc[spk_t['clusters']==clu_id, 'times'].values
            spk_arr[i, clu_spk_t - start_sample] = 1

        if clean_raster:
            spk_arr = sh.clean_spikeRaster_noisyEvents2d(spk_arr, verbose=verbose) # Remove noisy (simultaneous) events
            spk_arr = sh.remove_silent_channels_2d(spk_arr, verbose=verbose)

        return spk_arr
    
        
    def get_rasters_spikes(self, clu_list: np.array, start_samp_list: np.array, span_samples: int, clean_raster=False, verbose=True) -> np.array:
        """
        Return raster of spiking events for clusters of interest (clu_list) within a LIST OF WINDOWS of interest of the same length (start_sample
        -> start_sample+span_samples)

        Params:
            spk_df: dataframe containing ALL spikes specifying cluster, nucleus, spike-time, main channel and ksort label.
            clu_list: list of clusters to include in the raster plot
            start_samp_list: list of start samples of interest
            span_samples: length of the time window (number of samples)
        Keyword Params:
            clean_raster: bool. Remove events that co-occur across clusters

        Return: np.array of spiking events within the window of interest [m_clusters x n_timestamps] for each period of interest
        """

        # Build array of spiking events [m_clusters x n_timestamps] for the clusters of interes, for each period specified by the start_samp_list
        spk_arr_list = [self.get_window_spikes(clu_list, i, i+span_samples, clean_raster=clean_raster, verbose=verbose) for i in start_samp_list]
        return np.stack(spk_arr_list, axis=-1)


    def get_rasters_audio(self, start_sample_list: list, span_samples: int) -> list:
        """
        Return array of snippets of audio specified by start_sample_list (start_sample -> start_sample+span_samples)

        Params:
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
                print(n+' '+q+':', len(np.unique(self.clu_df.loc[(self.clu_df['quality']==q) & (self.clu_df['nucleus'].isin([n])),
                                                                 'cluster_id'])))
                
    
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
        Warning: If path = self.cluster_df_path, the cluster_df_path file will be OVERWRITTEN.
        """
        self.clu_df.to_pickle(save_path)