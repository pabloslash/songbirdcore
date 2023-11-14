from scipy import signal
import numpy as np 
from scipy.signal import butter, filtfilt
import librosa
from matplotlib import pyplot as plt


''' SPECTROGRAMS '''


def pretty_spectrogram(x, s_f, log=True, fft_size=512, step_size=64, window=None,
                       db_cut=65, f_min=0., f_max=None, plot=False, ax=None): 
    """
    Hello

    Parameters:
        x
        s_f
        log
        fft_size
        step_size
        window
        db_cut: db_cut=0 for no_trhesholding
        f_min
        f_max
        plot
        ax
    Output:

    """
    
    if window is None:
        # window = sg.windows.hann(fft_size, sym=False)
        window = ('tukey', 0.25)

    f, t, specgram = signal.spectrogram(x, fs=s_f, window=window,
                                       nperseg=fft_size,
                                       noverlap=fft_size - step_size,
                                       nfft=None,
                                       detrend='constant',
                                       return_onesided=True,
                                       scaling='spectrum',
                                       axis=-1)
                                       #mode='psd')

    if db_cut>0:
        specgram = spectrogram_db_cut(specgram, db_cut=db_cut, log_scale=False)
    if log:
        specgram = np.log10(specgram)

    if f_max is None:
        f_max = s_f/2.

    f_filter = np.where((f >= f_min) & (f < f_max))
    # return f, t, specgram

    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        ax.pcolormesh(t, f[f < f_max], specgram[f < f_max, :],
            cmap='inferno',
            rasterized=True)
        
        return f[f_filter], t, specgram[f_filter], ax

    return f[f_filter], t, specgram[f_filter]


def mel_spectrogram(sig, n_fft=1024, hop_length=128, fs_orig=40000):
    """
    Converts anaudio signal to MEL spectrogram.
    """

    # Spectrogram
    sgram = librosa.stft(sig.astype(float), n_fft=n_fft, hop_length=hop_length)

    # MEl spectrogram
    sgram_mag, _ = librosa.magphase(sgram)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=fs_orig)

    # Normalize MEL spectrogram
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

    # Remove DC and log scale
    # f = f[1:]

    return mel_sgram


def ms_spectrogram(x, s_f, n_window=512, step_ms=1, f_min=100, f_max=9000, cut_off=0.000055):

    # the overlap is the size of the window minus the samples in a msec
    msec_samples = int(s_f * 0.001)
    n_overlap = n_window - msec_samples * step_ms
    sigma = 1 / 200. * s_f

    # Make the spectrogram
    f, t, Sxx = signal.spectrogram(x, s_f,
                                   nperseg=n_window,
                                   noverlap=n_overlap,
                                   window=signal.gaussian(n_window, sigma),
                                   scaling='spectrum')

    if cut_off > 0:
        Sxx[Sxx < np.max((Sxx) * cut_off)] = 1
    
    Sxx[f<f_min, :] = 1

    return f[(f>f_min) & (f<f_max)], t, Sxx[(f>f_min) & (f<f_max)]


def rosa_spectrogram(y, hparams):
    D = rosa._stft(rosa.preemphasis(y,hparams), hparams)
    S = rosa._amp_to_db(np.abs(D)) - hparams['ref_level_db']
    return rosa._normalize(S, hparams)


def inv_spectrogram(spectrogram, hparams):
    """
    Converts spectrogram to waveform using librosa
    """
    S = rosa._db_to_amp(rosa._denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # Convert back to linear
    return rosa.inv_preemphasis(rosa._griffin_lim(S ** hparams['power'], hparams), hparams)          # Reconstruct phase


