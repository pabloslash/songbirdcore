from sklearn.decomposition import PCA
import quantities as pq
import neo
import numpy as np
import numpy.linalg as linalg
from .statespace_analysis_utils import compute_trajectories_dispersion


class PCACore:
    def __init__(self, neural_traces: np.ndarray, fs_neural: float, behavioral_traces: np.array = None, fs_behavior: float = None, labels: np.array = None, fs_labels: float
                 = None):
        """
        Initialize the PCACore class.

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

    
    def instantiate_pca(self, latent_dim: int):
        """
        Instantiate a new PCA model (sklearn.decomposition.PCA).

        :param latent_dim: Latent dimensionality of the PCA space
        """
        self.pca = PCA(n_components=latent_dim)

        
    def fit_transform_pca(self, spike_trains: np.ndarray) -> dict:
        """
        Obtain trajectories of neural activity in a low-dimensional latent space by trhough PCA.

        :param neo_spike_trains: Array [trials x neurons x spiketrains/rates]
        :return: Dictionary containing PCA model, trajectories, bin width, latent dimensions and more
        """

        trials, neurons, timesteps = spike_trains.shape

        # Transpose to [neurons x trials x timesteps]
        spike_trains = np.transpose(spike_trains, (1, 0, 2))
        # Flatten along last axis to to [neurons x trials*timesteps]
        spike_trains = np.reshape(spike_trains, [neurons, -1], order='C')

        # Fit PCA
        pcs = self.pca.fit_transform(spike_trains.T).T

        # Reshape back to [trials x neurons x timesteps]
        pcs = np.reshape(pcs, [self.pca.n_components, trials, -1], order='C').transpose(1, 0, 2)

        # Compute latent dispersion
        pcs_latent_dispersion = compute_trajectories_dispersion(pcs)

        # Build 'trajectories' dictionary:
        self.pca_dict = {
            'model': self.pca,
            'trajectories': pcs,
            # 'bin_w': self.gpfa.bin_size,
            'latent_dim': self.pca.n_components,
            'var_explained': sum(self.pca.explained_variance_ratio_),
            'trajectories_latent_dispersion': pcs_latent_dispersion,
            'latent_disp_mean': np.mean(pcs_latent_dispersion),
            'latent_disp_std': np.std(pcs_latent_dispersion)
        }
        return self.pca_dict

        