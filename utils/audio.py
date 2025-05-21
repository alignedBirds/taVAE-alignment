import os
import numpy as np
import soundfile as sf
from scipy import signal


def load_ogg(filepath, fs_resample):
    data, fs = sf.read(filepath)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if fs != fs_resample:
        num_samples = int(np.ceil(data.shape[0] * fs_resample / fs))
        data = signal.resample(data, num_samples)

    return data


def get_spectrogram_tensor(waveform, stft_params):
    if waveform.ndim == 1:
        waveform = waveform[:, np.newaxis]

    n_channels = waveform.shape[1]
    win_size = stft_params['window_size']
    win_shift = stft_params['window_shift']
    win_type = stft_params['type']

    frames_ = np.floor((waveform.shape[0] + 2 * win_shift) / win_shift)
    frames = int(np.ceil(frames_ / 8) * 8)

    X = np.zeros((int(win_size / 2) + 1, int(frames), n_channels), dtype=np.complex64)

    for ch in range(n_channels):
        f, t, Z = signal.stft(waveform[:, ch], fs=stft_params['fs'],
                              nperseg=win_size, noverlap=win_size - win_shift, window=win_type)
        X[:, :Z.shape[1], ch] = Z

    return X