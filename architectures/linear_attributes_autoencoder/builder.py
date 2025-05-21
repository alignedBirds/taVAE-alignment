from framework.architectures import ImageVAE
from .encoder import LinearEncoder
from .decoder import LinearDecoder

from typing import Optional


class Builder():
    def build(self, latent_dim: int,
              dims: Optional[list[int]] = [32, 64, 128]
            ) -> ImageVAE:
        
        encoder = LinearEncoder(
            dims,
            latent_dim
        )

        decoder = LinearDecoder(
            dims[::-1],
            latent_dim
        )

        return ImageVAE(encoder, decoder)