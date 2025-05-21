import torch
import torch.nn as nn
from framework.variational_autoencoder import Encoder
from framework import Tensor, Mu, Sigma
from typing import Tuple


class LinearEncoder(Encoder, nn.Module):
    def __init__(self, dims, latent_dim):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(inplace=True)
            ))

        self.layers = nn.Sequential(*layers)

        self.mu_proj = nn.Linear(dims[-1], latent_dim)
        self.logvar_proj = nn.Linear(dims[-1], latent_dim)

    def forward(self, x: Tensor) -> Tuple[Mu, Sigma]:
        h = self.layers(x)
        mu = self.mu_proj(h)
        logvar = torch.clamp(self.logvar_proj(h), min=-10, max=10)
        return mu, logvar