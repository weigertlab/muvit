import logging
import sys
from abc import ABC
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import yaml
from torch import Tensor
from x_transformers.x_transformers import Attention, FeedForward, RotaryEmbedding

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


class RotaryEmbeddingTrainable(RotaryEmbedding):
    def __init__(self, dim: int, base: int = 256):
        super().__init__(dim=dim, base=base)
        self.inv_freq = nn.Parameter(self.inv_freq)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        rotary_dim: Optional[int],
        heads: int = 4,
        rotary_base: int = 512,
        dropout: float = 0.0,
    ):
        """Initialize a transformer layer with attention and feed-forward components.
           Option for rotary positional embeddings.

        Args:
            dim: Dimension of input/output tensors
            heads: Number of attention heads
            rotary_dim: Dimension for rotary embeddings, or None to disable
            dropout: Dropout probability for attention and feed-forward layers
        """
        super().__init__()
        self.attn = Attention(
            dim=dim, heads=heads, dim_head=dim // heads, flash=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim, mult=2, dropout=dropout)

        if rotary_dim is not None:
            dims = [dim // heads // rotary_dim] * (rotary_dim - 1)
            dims = [2 * (d // 2) for d in dims]
            dims = [dim // heads - sum(dims)] + dims
            assert min(dims) > 8, "Rotary embedding dimension must be greater than 8"
            self.rotary_pos_embs = nn.ModuleList(
                [RotaryEmbeddingTrainable(dim=d, base=rotary_base) for d in dims]
            )
        else:
            self.rotary_pos_embs = None

    def forward(
        self,
        x: Tensor,
        level_idx: Tensor = None,
        coords: Tensor = None,
        attention_mode: Literal["all", "causal", "same", "random"] = "all",
        context: Tensor = None,
        context_level_idx: Tensor = None,
        context_coords: Tensor = None,
    ):
        """Apply transformer layer with optional rotary embeddings and level-based attention.

        If context is provided, performs cross-attention between x and context. Otherwise performs self-attention.
        Supports rotary embeddings and level-based attention masking.

        Args:
            x: Input tensor of shape (B, N, D)
            level_idx: Level indices for each token
            coords: Coordinates for rotary embeddings
            attention_mode: Type of attention masking ('all', 'causal', 'same', 'random')
            context: Optional context tensor for cross-attention. If provided, performs cross-attention between x and context
            context_level_idx: Level indices for context tokens
            context_coords: Coordinates for context rotary embeddings

        Returns:
            Transformed tensor of same shape as input
        """
        B, N, D = x.shape

        rotary_pos_emb = None
        if coords is not None and self.rotary_pos_embs is not None:
            assert coords.shape[:2] == x.shape[:2]
            freqs, scales = zip(
                *[r(coords[..., i]) for i, r in enumerate(self.rotary_pos_embs)]
            )
            freqs = torch.cat(freqs, dim=-1).contiguous()
            scales = scales[0]
            rotary_pos_emb = freqs, scales

        if context_coords is not None and self.rotary_pos_embs is not None:
            assert context_coords.shape[:2] == context.shape[:2]
            freqs, scales = zip(
                *[r(context_coords[..., i]) for i, r in enumerate(self.rotary_pos_embs)]
            )
            freqs = torch.cat(freqs, dim=-1).contiguous()
            scales = scales[0]
            context_rotary_pos_emb = freqs, scales
        else:
            context_rotary_pos_emb = None

        if context is not None:
            assert context.shape[0] == B
            assert context.shape[2] == D
            NC = context.shape[1]
        else:
            NC = N

        if context_level_idx is None:
            context_level_idx = level_idx

        if level_idx is not None:
            if attention_mode == "all":
                mask = None
            elif attention_mode == "causal":
                mask = level_idx.view(B, 1, N, 1) >= context_level_idx.view(B, 1, 1, NC)
            elif attention_mode == "same":
                mask = level_idx.view(B, 1, NC, 1) == context_level_idx.view(
                    B, 1, 1, NC
                )
            elif attention_mode == "random":
                # Create random boolean mask for each batch and head
                mask = torch.argsort(torch.rand(N, NC, device=x.device), dim=1) == 0
            else:
                raise ValueError(f"Invalid attention mode: {attention_mode}")
        else:
            mask = None

        x = x + self.attn(
            self.norm1(x),
            attn_mask=mask,
            rotary_pos_emb=rotary_pos_emb,
            context=context,
            context_rotary_pos_emb=context_rotary_pos_emb,
        )
        x = x + self.ff(self.norm2(x))
        return x


class SaveableModel(nn.Module, ABC):
    """Base class for models that can be saved and loaded from disk."""

    def __init__(self):
        super().__init__()
        self._config = {}

    def save(self, path: str | Path, overwrite: bool = False):
        """Save model configuration and state to a folder.

        Args:
            path: Path to save the model
            overwrite: Whether to overwrite existing files
        """
        log.info(f"Saving model to {path}")
        path = Path(path)
        if path.exists() and not overwrite:
            raise ValueError(
                f"Path already exists: {path} (set overwrite=True to overwrite)"
            )
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.yaml", "w") as f:
            yaml.dump(self._config, f)
        torch.save(self.state_dict(), path / "model.pth")

    @classmethod
    def from_folder(cls, path: str | Path, device: str | torch.device = "cpu"):
        """Load model from a saved folder.

        Args:
            path: Path to the saved model folder
            device: Device to load the model on

        Returns:
            Loaded model instance
        """
        path = Path(path)
        with open(path / "config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = cls(**config).to(device)
        model.load_state_dict(torch.load(path / "model.pth", map_location=device))
        return model
