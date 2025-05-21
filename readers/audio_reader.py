from typing import Optional, Callable, Tuple
from torch.utils.data import Dataset
import torch

import os
from utils.audio import load_ogg, get_spectrogram_tensor


class Reader(Dataset):
    def __init__(self, fourier_params, len = 100):
        self.dir = "data/raw_data/train_soundscapes"
        self.len = len
        self.fourier_params = fourier_params

        self.audio_files = os.listdir(self.dir)
        self.waves = self.__load_waves()
    
    def __len__(self):
        return min(self.len, len(self.audio_files))

    def __getitem__(self, idx):
        return self.waves[idx]
    
    def __load_waves(self):
        waves = []
        for audio in self.audio_files[:self.len]:
            audio = load_ogg(f'{self.dir}/{audio}', self.fourier_params['fs'])
            waves.append(get_spectrogram_tensor(audio, self.fourier_params))
        return waves