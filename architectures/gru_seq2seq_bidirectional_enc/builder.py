from framework.architectures import ImageVAE
import torch.nn as nn

from .encoder import BidirectionalEncoder
from .decoder import AutoregressiveDecoder

from typing import Optional


class Builder():
    def build(self,
              embedding_dim: int,
              latent_dim: int,
              context_length: int,
              num_layers: Optional[int] = 1
            ) -> ImageVAE:
        
        encoder = BidirectionalEncoder(
            embedding_dim,
            latent_dim,
            num_layers
          )

        decoder = AutoregressiveDecoder(
            embedding_dim,
            latent_dim,
            context_length,
            num_layers
          )
        
        return ImageVAE(encoder, decoder)