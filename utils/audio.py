import os
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
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

    X = torch.tensor(X)
    X = X.squeeze(-1)
    X = X.permute(1, 0)
    return X.unsqueeze(0) 


def concat_FT(X):
    real = X.real
    imag = X.imag

    return torch.cat([real, imag], dim=-1)


def reverse_FT(Y):
    half = Y.shape[-1] // 2
    real = Y[..., :half]
    imag = Y[..., half:]
    
    return torch.complex(real, imag)


def play_audio(waveform, sample_rate=16000, duration=5):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    if waveform.ndim > 1:
        waveform = np.squeeze(waveform)
    
    max_samples = sample_rate * duration
    waveform = waveform[:max_samples]

    waveform = waveform.astype(np.float32)

    sd.play(waveform, samplerate=sample_rate)
    sd.wait()


def get_waveform_from_spectrogram_tensor(X: torch.Tensor, stft_params: dict) -> np.ndarray:
    X = X.squeeze(0).permute(1, 0)
    X_np = X.numpy()

    _, waveform = signal.istft(X_np,
                               fs=stft_params['fs'],
                               nperseg=stft_params['window_size'],
                               noverlap=stft_params['window_size'] - stft_params['window_shift'],
                               window=stft_params['type'])
    
    return waveform