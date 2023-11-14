import quantities as pq
import neo
import numpy as np
import deprecated


def convert_to_neo_spike_trains(spikes_array: np.ndarray, fs_neural: float):
    """
    Convert a binary (0s, 1s) array of spikes [neurons x spiketrains] to a list of neo.SpikeTrains.

    :param spikes_array: Binary array of spikes [neurons x spiketrains]
    :param fs_neural: Sampling frequency of neural traces
    :return: List of neo.SpikeTrain objects
    """
    t_stop = spikes_array.shape[1] / fs_neural
    return [neo.SpikeTrain(times=np.where(x != 0)[0] / fs_neural, units='sec', t_stop=t_stop) for x in spikes_array]


def convert_to_neo_spike_trains_3d(spikes_array_3d: np.ndarray, fs_neural: float):
    """
    Convert a binary (0s, 1s) array of arrays of spikes [trials x neurons x spiketrains] to a list of lists of neo.SpikeTrains.

    :param spikes_array_3d: Binary array of arrays of spikes [trials x neurons x spiketrains]
    :param fs_neural: Sampling frequency of neural traces
    :return: List of lists of neo.SpikeTrain objects
    """
    return [convert_to_neo_spike_trains(spikes_array_3d[t], fs_neural=fs_neural) for t in range(spikes_array_3d.shape[0])]


def compute_trajectories_dispersion(trajectories: np.ndarray):
    """
    Compute the latent dispersion at each timestep around the mean trajectory.

    :param trajectories: np.array [trials x dimensions x timesteps]
    :return: np.array containing latent dispersion
    """

    # Mean trajectory across time (u_t)
    mean_trajectory = np.mean(trajectories, 0)

    # Mean of spanned latent space (u)
    mean_latent_space = np.mean(mean_trajectory, 1)

    # Calculate total absolute variance around the mean of the latent space (x_t - u)
    trials, dimensions, time = trajectories.shape
    array_like_mean_latent_space = np.tile(np.tile(mean_latent_space, (time, 1)).T, (trials, 1, 1))
    var_total = np.mean(np.abs(trajectories - array_like_mean_latent_space))

    # Calculate the dispersion around the mean trajectory (x_t - u_t)
    array_like_mean_trajectory = np.tile(mean_trajectory, (trials, 1, 1))
    latent_dispersion = np.mean(np.abs(trajectories - array_like_mean_trajectory))

    relative_latent_dispersion = latent_dispersion / var_total

    return relative_latent_dispersion


def permute_array_rows_independently(array: np.ndarray) -> np.ndarray:
    """
        Shuffle column indexes for each row of an ndarray independently (sum over rows is conserved).
        In an array of spiketrains [neurons x spiketrains], it breaks the temporal correlation of each neuron spiketimes, while maintaining the individual neuron overall spikerates.
        
        e.g.
            Original Array:
        [[1 2 3]
         [4 5 6]
         [7 8 9]]

        Permuted Array (Row-wise):
        [[1 3 2]
         [5 6 4]
         [8 7 9]]
    """
    num_rows, num_cols = array.shape
    permuted_array = np.zeros((num_rows, num_cols), dtype=array.dtype)

    for i in range(num_rows):
        permutation_indices = np.random.permutation(num_cols)
        permuted_array[i,:] = array[i, permutation_indices]

    assert (np.sum(array, axis=1) == np.sum(permuted_array, axis=1)).all(), 'WARNING: Sum over rows in orginal and permuted array does not match. Check permutation!'
    
    return permuted_array


def permute_array_cols_independently(array: np.ndarray) -> np.ndarray:
    """
        Shuffle row indexes for each row of an ndarray independently (sum over colums is conserved).
        In an array of spiketrains [neurons x spiketrains], it breaks the structure within each neural trace, while maintaining the network's instantaneous spikerate.
        
        e.g.
            Original Array:
        [[1 2 3]
         [4 5 6]
         [7 8 9]]

        Permuted Array (Row-wise):
        [[4 8 6]
         [1 2 9]
         [7 5 3]]
    """
    num_rows, num_cols = array.shape
    permuted_array = np.zeros((num_rows, num_cols), dtype=array.dtype)

    for i in range(num_cols):
        permutation_indices = np.random.permutation(num_rows)
        permuted_array[:,i] = array[permutation_indices, i]

    assert (np.sum(array, axis=0) == np.sum(permuted_array, axis=0)).all(), 'WARNING: Sum over columns in orginal and permuted array does not match. Check permutation!'
    
    return permuted_array


@deprecated.deprecated(reason="This method is deprecated. Use compute_trajectories_dispersion() instead.")
def compute_trajectories_dispersion_legacy(trajectories: np.ndarray) -> np.ndarray:
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