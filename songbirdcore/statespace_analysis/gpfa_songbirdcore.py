from elephant.gpfa import GPFA
import quantities as pq
import neo
import numpy as np
import numpy.linalg as linalg
from .statespace_analysis_utils import compute_trajectories_dispersion


class GPFACore:
    
    def __init__(self, neural_traces: np.ndarray, fs_neural: float, behavioral_traces: np.array=None, fs_behavior: float=None, labels: np.array=None, fs_labels: float=None):
        """
        Initialize the GPFACore class.

        :param neural_traces: Array of neural traces ([trials x neurons x spiketrains])
        :param fs_neural: Sampling frequency of neural traces
        :param behavioral_traces: Array of behavioral trials corresponding to neural traces [trials x behavioral_traces]
        :param fs_behavior: Sampling frequency of behavioral traces
        :param labels: Array of behavioral labels corresponding to neural/behavioral trials [trials x labels]
        :param fs_labels: Sampling frequency of labels
        """
        self.neural_traces = neural_traces
        self.fs_neural = fs_neural

        if behavioral_traces is not None:
            if fs_behavior is None:
                raise ValueError('Expected argument "fs_behavior" for behavioral sampling frequency')
            self.fs_behavior = fs_behavior
            self.behavioral_traces = behavioral_traces

        if labels is not None:
            if fs_labels is None:
                raise ValueError('Expected argument "fs_labels" for labels sampling frequency')
            self.labels = labels
            self.fs_labels = fs_labels
    
            
    def set_behavioral_traces(self, behavioral_traces: np.array, fs_behavior: float):
        """
        Set behavioral traces and their sampling frequency.

        :param behavioral_traces: Array of behavioral trials corresponding to neural traces [trials x behavioral_traces]
        :param fs_behavior: Sampling frequency of behavioral traces
        """
        self.fs_behavior = fs_behavior
        self.behavioral_traces = behavioral_traces

        
    def set_label_traces(self, labels: np.array, fs_labels: float):
        """
        Set behavioral labels and their sampling frequency.

        :param labels: Array of behavioral labels corresponding to neural/behavioral trials [trials x labels]
        :param fs_labels: Sampling frequency of labels
        """
        self.labels = labels
        self.fs_labels = fs_labels

    
    def instantiate_gpfa(self, bin_size: pq.quantity.Quantity, latent_dim: int, em_max_iters: int=None):
        """
        Instantiate a new GPFA model (elephant.gpfa).

        :param bin_size: Time bin size (e.g., 15 * pq.ms)
        :param latent_dim: Latent dimensionality of the GPFA space
        """
        # GPFA params
        em_max_iters = em_max_iters if em_max_iters else 500
        
        self.gpfa = GPFA(bin_size=bin_size, x_dim=latent_dim, em_max_iters=em_max_iters)

        
    def fit_transform_gpfa(self, neo_spike_trains: list):
        """
        Obtain trajectories of neural activity in a low-dimensional latent variable space by inferring the posterior mean of the GPFA model and applying
        orthonormalization on the latent variable space.

        :param neo_spike_trains: List [trials x neurons x neo.core.spiketrain.SpikeTrain]
        :return: Dictionary containing GPFA model, trajectories, bin width, latent dimensions and more
        """
        
        trajectories =  np.stack(self.gpfa.fit_transform(neo_spike_trains), axis=0)
        trajectories_latent_dispersion = compute_trajectories_dispersion(trajectories)

        self.gpfa_dict = {
            'model': self.gpfa,
            'trajectories': trajectories,
            'bin_w': self.gpfa.bin_size,
            'latent_dim': self.gpfa.x_dim,
            'var_explained': self.compute_gpfa_variance_explained(),
            'trajectories_latent_dispersion': trajectories_latent_dispersion#,
            # 'latent_disp_mean': np.mean(trajectories_latent_dispersion),
            # 'latent_disp_std': np.std(trajectories_latent_dispersion)
        }
        return self.gpfa_dict

    
    def compute_gpfa_variance_explained(self):
        """
        Compute GPFA variance explained.

        :return: Dictionary containing GPFA variance explained
        """
        
        C = self.gpfa.params_estimated['C'] # Loading matrix
        R = self.gpfa.params_estimated['R'] # Noise matrix

        total_var = np.trace( np.dot(C, C.transpose()) + R )

        # variance explained
        shared_var = np.trace(np.dot(C, C.transpose())) / total_var;
        private_var = np.trace(R) / total_var;
        
        return shared_var
    