import torch.nn as nn
import torch.nn.functional as F
from framework import LatentTensor, Tensor
from framework import Decoder

class LinearDecoder(Decoder, nn.Module):
    def __init__(self, 
                 dims: list[int],
                 latent_dim: int):
        
        super().__init__()

        self.latent_proj = nn.Linear(latent_dim, dims[0])
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True) if i < len(dims) - 1 else nn.Sigmoid()
            ))

        self.decoder_blocks = nn.Sequential(*layers)

    def forward(self, z: LatentTensor) -> Tensor:
        h = self.latent_proj(z)
        return self.decoder_blocks(h)