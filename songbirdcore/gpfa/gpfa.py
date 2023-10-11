from elephant.gpfa import GPFA
import quantities as pq
import neo
import numpy as np
import numpy.linalg as linalg


class GPFACore:
    def __init__(self, neural_traces: dict, fs_neural: float, behavioral_traces: np.array = None, fs_behavior: float = None, labels: np.array = None, fs_labels: float
                 = None):
        """
        Initialize the GPFACore class.

        :param neural_traces: Dictionary of neural traces (key: [motifs x neurons x spiketrains])
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

            
    def add_behavioral_traces(self, behavioral_traces: np.array, fs_behavior: float):
        """
        Set behavioral traces and their sampling frequency.

        :param behavioral_traces: Array of behavioral trials corresponding to neural traces [trials x behavioral_traces]
        :param fs_behavior: Sampling frequency of behavioral traces
        """
        self.fs_behavior = fs_behavior
        self.behavioral_traces = behavioral_traces

        
    def add_label_traces(self, labels: np.array, fs_labels: float):
        """
        Set behavioral labels and their sampling frequency.

        :param labels: Array of behavioral labels corresponding to neural/behavioral trials [trials x labels]
        :param fs_labels: Sampling frequency of labels
        """
        self.labels = labels
        self.fs_labels = fs_labels

        
    @staticmethod
    def convert_to_neo_spike_trains(spikes_array, fs_neural):
        """
        Convert a binary (0s, 1s) array of spikes [neurons x spiketrains] to a list of neo.SpikeTrains.

        :param spikes_array: Binary array of spikes [neurons x spiketrains]
        :param fs_neural: Sampling frequency of neural traces
        :return: List of neo.SpikeTrain objects
        """
        t_stop = spikes_array.shape[1] / fs_neural
        return [neo.SpikeTrain(times=np.where(x != 0)[0] / fs_neural, units='sec', t_stop=t_stop) for x in spikes_array]

    
    @staticmethod
    def convert_to_neo_spike_trains_3d(spikes_array_3d, fs_neural):
        """
        Convert a binary (0s, 1s) array of arrays of spikes [trials x neurons x spiketrains] to a list of lists of neo.SpikeTrains.

        :param spikes_array_3d: Binary array of arrays of spikes [trials x neurons x spiketrains]
        :param fs_neural: Sampling frequency of neural traces
        :return: List of lists of neo.SpikeTrain objects
        """
        return [GPFACore.convert_to_neo_spike_trains(spikes_array_3d[t], fs_neural=int(fs_neural)) for t in range(spikes_array_3d.shape[0])]

    
    def instantiate_gpfa(self, bin_size: pq.quantity.Quantity, latent_dim: int):
        """
        Instantiate a new GPFA model (elephant.gpfa).

        :param bin_size: Time bin size (e.g., 15 * pq.ms)
        :param latent_dim: Latent dimensionality of the GPFA space
        """
        self.gpfa = GPFA(bin_size=bin_size, x_dim=latent_dim)

        
    def fit_transform_gpfa(self, neo_spike_trains: list):
        """
        Obtain trajectories of neural activity in a low-dimensional latent variable space by inferring the posterior mean of the GPFA model and applying
        orthonormalization on the latent variable space.

        :param neo_spike_trains: List [trials x neurons x neo.core.spiketrain.SpikeTrain]
        :return: Dictionary containing GPFA model, trajectories, bin width, and latent dimension
        """
        trajectories =  np.stack(self.gpfa.fit_transform(neo_spike_trains), axis=0)
        trajectories_latent_dispersion = self.compute_trajectories_dispersion(trajectories)

        self.gpfa_dict = {
            'model': self.gpfa,
            'trajectories': trajectories,
            'bin_w': self.gpfa.bin_size,
            'latent_dim': self.gpfa.x_dim,
            'var_explained': self.compute_gpfa_variance_explained(),
            'trajectories_latent_dispersion': trajectories_latent_dispersion,
            'latent_disp_mean': np.mean(trajectories_latent_dispersion),
            'latent_disp_std': np.std(trajectories_latent_dispersion)
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
    
    
    @staticmethod
    def compute_trajectories_dispersion(trajectories):
        """
        Compute the latent dispersion at each timestep around the mean trajectory.
        
        :param trajectories: np.array [trials x dimensions x timesteps]
        :return: np.array containing latent dispersion
        """
    
        # Mean trajectory across time
        mean_trajectory = np.mean(trajectories, 0)

        # Mean of spanned latent space
        mean_latent_space = np.mean(mean_trajectory, 1)

        latent_dispersion = []
        for t in range(mean_trajectory.shape[1]):

            mean_trajectory_t = mean_trajectory[:, t]
            trial_dispersion_t = np.linalg.norm((trajectories[:, :, t] - mean_trajectory_t), ord=None, axis=1) 

            # Normalize dispersion by 2-norm of mean_trajectory to mean_latent_space at time (t) to compare across spaces
            # trajectory_normalizer_t = np.linalg.norm(mean_trajectory_t - mean_latent_space, ord=None) # [scalar] 
            trajectory_normalizer_t = np.linalg.norm(mean_latent_space, ord=None) # [scalar] 
            normalized_trial_dispersion_t = trial_dispersion_t / trajectory_normalizer_t # [1 x trials]

            # Average trial dispersion at time (t)
            latent_dispersion.extend([np.mean(normalized_trial_dispersion_t)])

        return latent_dispersion  # final output [1 x timesteps]