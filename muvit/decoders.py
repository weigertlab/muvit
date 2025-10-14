from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, Tuple, TypeVar

import torch.nn as nn
from torch import Tensor
from x_transformers.x_transformers import RotaryEmbedding

from .bblocks import SaveableModel, TransformerLayer, RotaryEmbeddingTrainable

T = TypeVar("T", bound=Tuple[int, ...])

class MuViTDecoder(SaveableModel, ABC, Generic[T]):
    def __init__(
        self,
        in_channels: int,
        dim: int = 320,
        num_layers: int = 2,
        heads: int = 4,
        rotary_mode: Literal["none", "fixed", "shared", "per_layer"] = "per_layer",
        rotary_base: int = 10000,
        dropout: float = 0.0,
        use_rotary_embed: Optional[bool] = None,
    ):
        """Initialize a Vision Transformer decoder.

        Args:
            in_channels: Number of input channels
            dim: Hidden dimension
            num_layers: Number of transformer layers
            heads: Number of attention heads
            rotary_mode: Type of rotary embeddings (none/fixed/shared/per_layer)
            rotary_base: Base for rotary embeddings
            dropout: Dropout probability for transformer layers
            use_rotary_embed: Deprecated, use rotary_mode instead
        """
        super().__init__()

        if use_rotary_embed is not None:
            import warnings
            warnings.warn(
                "use_rotary_embed is deprecated, use rotary_mode instead",
                DeprecationWarning,
                stacklevel=2,
            )
            rotary_mode = "per_layer" if use_rotary_embed else "none"

        self.proj = nn.Sequential(
            nn.Linear(in_channels, dim), nn.GELU(), nn.LayerNorm(dim)
        )

        shared_rope = None
        per_layer_rotary_dim = None

        if rotary_mode != "none":
            dims = [dim // heads // self.ndim] * (self.ndim - 1)
            dims = [2 * (d // 2) for d in dims]
            dims = [dim // heads - sum(dims)] + dims
            assert min(dims) > 8, "Rotary embedding dimension must be greater than 8"

            if rotary_mode == "fixed":
                shared_rope = nn.ModuleList([
                    RotaryEmbedding(dim=d, base=rotary_base) for d in dims
                ])
            elif rotary_mode == "shared":
                shared_rope = nn.ModuleList([
                    RotaryEmbeddingTrainable(dim=d, base=rotary_base) for d in dims
                ])
            elif rotary_mode == "per_layer":
                per_layer_rotary_dim = self.ndim

        if shared_rope is not None:
            self.rotary_pos_embs = shared_rope

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    dim=dim,
                    heads=heads,
                    rotary_dim=per_layer_rotary_dim,
                    rotary_pos_embs=shared_rope,
                    rotary_base=rotary_base,
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
    
    @classmethod
    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of spatial dimensions"""
        pass


class MuViTDecoder2d(MuViTDecoder[Tuple[int, int]]):
    @classmethod
    @property
    def ndim(self) -> int: # FIXME: deprecated in py313
        return 2

class MuViTDecoder3d(MuViTDecoder[Tuple[int, int, int]]):
    @classmethod
    @property
    def ndim(self) -> int: # FIXME: deprecated in py313
        return 3

class MuViTDecoder4d(MuViTDecoder[Tuple[int, int, int]]):
    @classmethod
    @property
    def ndim(self) -> int: # FIXME: deprecated in py313
        return 4
