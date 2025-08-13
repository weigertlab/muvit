from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import torch.nn as nn
from torch import Tensor

from .bblocks import SaveableModel, TransformerLayer

T = TypeVar("T", bound=Tuple[int, ...])

class MuViTDecoder(SaveableModel, ABC, Generic[T]):
    def __init__(
        self,
        in_channels: int,
        dim: int = 320,
        num_layers: int = 2,
        heads: int = 4,
        use_rotary_embed: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize a Vision Transformer decoder.

        Args:
            in_channels: Number of input channels
            dim: Hidden dimension
            num_layers: Number of transformer layers
            heads: Number of attention heads
            use_rotary_embed: Whether to use rotary embeddings
            ndim: Number of spatial dimensions
            dropout: Dropout probability for transformer layers
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, dim), nn.GELU(), nn.LayerNorm(dim)
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=dim,
                    heads=heads,
                    rotary_dim=self.ndim if use_rotary_embed else None,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dim = dim

    def forward(
        self,
        x: Tensor,
        coords: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_coords: Optional[Tensor] = None,
    ):
        """Decode features with optional cross-attention.

        Args:
            x: Input features of shape (B, N, D)
            coords: Coordinates for rotary embeddings
            context: Optional context for cross-attention
            context_coords: Coordinates for context rotary embeddings

        Returns:
            Decoded features of same shape as input
        """
        B, _, D = x.shape
        x = self.proj(x)
        if context is not None:
            context = self.proj(context)

        for i, layer in enumerate(self.layers):
            # use cross attention for first layer
            if i == 0:
                x = layer(
                    x, coords=coords, context=context, context_coords=context_coords
                )
            else:
                x = layer(x, coords=coords)
        return x
    
    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of spatial dimensions"""
        pass


class MuViTDecoder2d(MuViTDecoder[Tuple[int, int]]):
    @property
    def ndim(self) -> int:
        return 2

class MuViTDecoder3d(MuViTDecoder[Tuple[int, int, int]]):
    @property
    def ndim(self) -> int:
        return 3
