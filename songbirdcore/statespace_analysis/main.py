import argparse
import pickle as pkl
import numpy as np
import quantities as pq

from songbirdcore.statespace_analysis.gpfa_songbirdcore import GPFACore
from songbirdcore.statespace_analysis.pca_songbirdcore import PCACore
from songbirdcore.statespace_analysis.statespace_analysis_utils import convert_to_neo_spike_trains, convert_to_neo_spike_trains_3d
import songbirdcore.spikefinder.spike_analysis_helper as sh
from songbirdcore.params import GlobalParams as gparams
from songbirdcore.utils.data_utils import save_dataset


def main(file_path, neural_groups, bin_size_ms, latent_dim, neural_samp_perc, output_dir):
    
    bin_size = bin_size_ms * pq.ms

    # Load data
    with open(file_path, 'rb') as pickle_file:
        state_space_analysis_dict = pkl.load(pickle_file)

    print("Loaded Dictionary!")
    print(state_space_analysis_dict.keys())

    neural_dict = state_space_analysis_dict['neural_dict']
    audio_motifs = state_space_analysis_dict['audio_motifs']
    audio_labels = state_space_analysis_dict['audio_labels']
    fs_neural = state_space_analysis_dict['fs_neural']
    fs_audio = state_space_analysis_dict['fs_audio']
    t_pre = state_space_analysis_dict['t_pre']
    t_post = state_space_analysis_dict['t_post']
    sess_params = state_space_analysis_dict['sess_params']

    # Drop silent clusters
    for k in neural_dict.keys():
        num_trials = len(neural_dict[k])
        neural_dict[k] = np.delete(neural_dict[k], np.where(np.sum(neural_dict[k], axis=(0,2)) < num_trials)[0], axis=1)
        print(k, neural_dict[k].shape)

    # Generate spiketrains for shuffle control conditions
    for k in neural_groups:
        neural_dict[k+'_shuffle_time'] = np.array([sh.permute_array_rows_independently(i) for i in neural_dict[k]])
        neural_dict[k+'_shuffle_neurons'] = np.array([sh.permute_array_cols_independently(i) for i in neural_dict[k]])

    # State-Space Analysis
    state_space_analysis_results = {}
    resampled_neural_dict = {}
    
    for ng in neural_groups:
        neural_traces = neural_dict[ng]
        print(f'Processing {ng}')

        # Randomly sample neural channels
        if neural_samp_perc < 1:
            num_channels = neural_traces.shape[1]
            num_to_sample = round(num_channels * neural_samp_perc)
            sampled_indices = np.random.choice(num_channels, num_to_sample, replace=False)
            neural_traces = neural_traces[:, sampled_indices, :]
            print(ng, neural_traces.shape)
        resampled_neural_dict[ng] = neural_traces

        trajectories_dict = {k: {} for k in ['pca', 'gpfa']}
        for ld in [latent_dim]:
            if ld > neural_traces.shape[1]:
                print(f'ld: {ld}, clusters: {neural_traces.shape[1]}. ld > num_clusters: skipping state-space analysis.')
                continue

            # Fit PCA
            myPCA = PCACore(neural_traces, round(fs_neural), audio_motifs, fs_audio, audio_labels, fs_audio)
            myPCA.instantiate_pca(ld)
            spike_trains = sh.downsample_list_3d(myPCA.neural_traces, number_bin_samples=int(bin_size/1000*fs_neural), mode='sum')
            pca_dict = myPCA.fit_transform_pca(spike_trains)
            pca_dict['bin_w'] = bin_size
            trajectories_dict['pca'][ng+'_dim'+str(ld)] = pca_dict

            # Fit GPFA
            myGPFA = GPFACore(neural_traces, round(fs_neural), audio_motifs, fs_audio, audio_labels, fs_audio)
            myGPFA.instantiate_gpfa(bin_size, ld, em_max_iters=gparams.gpfa_max_iter)
            spike_trains = convert_to_neo_spike_trains_3d(myGPFA.neural_traces, myGPFA.fs_neural)
            gpfa_dict = myGPFA.fit_transform_gpfa(spike_trains)
            trajectories_dict['gpfa'][ng+'_dim'+str(ld)] = gpfa_dict

        state_space_analysis_results[ng] = trajectories_dict

        # Save data
        filename_appendix = f"{neural_samp_perc*100}%_neural_channels.pkl"
        save_dataset(state_space_analysis_results, 
                     audio_motifs, 
                     audio_labels, 
                     fs_neural, 
                     fs_audio, 
                     t_pre, t_post, 
                     sess_params, 
                     'RAW', 
                     output_dir, 
                     filename_appendix)
        
        save_dataset(state_space_analysis_results, 
                     audio_motifs, 
                     audio_labels, 
                     1/int(bin_size)*1000, 
                     fs_audio, 
                     t_pre, t_post, 
                     sess_params, 
                     'TRAJECTORIES', 
                     output_dir, 
                     filename_appendix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run state-space analysis on bird song data.")
    parser.add_argument("file_path", type=str, help="Path to the pickle file containing the data.")
    parser.add_argument("--neural_groups", nargs='+', default=['ra_all', 'hvc_all'], help="List of neural groups to analyze.")
    parser.add_argument("--bin_size_ms", type=int, default=5, help="Bin size in milliseconds for state-space analysis.")
    parser.add_argument("--latent_dim", type=int, default=12, help="Number of latent dimensions for PCA/GPFA.")
    parser.add_argument("--neural_samp_perc", type=float, default=0.5, help="Percentage of neural channels to sample.")
    parser.add_argument("--output_dir", type=str, default='./data', help="Directory to save the output files.")

    args = parser.parse_args()
    main(args.file_path, args.neural_groups, args.bin_size_ms, args.latent_dim, args.neural_samp_perc, args.output_dir)



