from framework import LatentTensor, Tensor
from framework import Decoder

import torch
import torch.nn as nn

from typing import Optional


class AutoregressiveDecoder(Decoder, nn.Module):
    def __init__(self, 
                 embedding_dim: int,
                 latent_dim: int,
                 context_length: int,
                 num_layers: Optional[int] = 1):

        super(AutoregressiveDecoder, self).__init__()

        self.proj_h = nn.Linear(latent_dim, embedding_dim)

        self.decoder = nn.GRU(
            embedding_dim, 
            embedding_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False
        )

        self.context_length = context_length

    def forward(self, 
                z: LatentTensor, **kwargs) -> Tensor:
        
        h = self.proj_h(z).unsqueeze(0)

        x = None

        outputs = []

        for _ in range(self.context_length):
            x, h = self.step(h, x)
            outputs.append(x.squeeze(1))

        return torch.stack(outputs, dim=1)
    
    def step(self,
             h: Tensor,
             x: Optional[Tensor] = None):

        if x == None:
            x = h.transpose(0, 1)

        x, h = self.decoder(x, h)
        return x, h