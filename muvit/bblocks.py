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


def _split_rope_dims(dim: int, heads: int, ndim: int) -> list[int]:
    """Split head dimension for multi-dimensional RoPE embeddings."""
    dims = [dim // heads // ndim] * (ndim - 1)
    dims = [2 * (d // 2) for d in dims]  # Ensure even
    dims = [dim // heads - sum(dims)] + dims  # Residual in first dim
    assert min(dims) > 8, f"RoPE dimension too small: min={min(dims)}, need >8"
    return dims


def compute_rotary_emb(coords: Tensor, rotary_embs: nn.ModuleList) -> tuple[Tensor, Tensor]:
    """Compute rotary embeddings from coordinates."""
    freqs, scales = zip(*[r(coords[..., i]) for i, r in enumerate(rotary_embs)])
    freqs = torch.cat(freqs, dim=-1).contiguous()
    scales = scales[0]
    return freqs, scales


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        rotary_dim: Optional[int],
        heads: int = 4,
        rotary_base: int = 10000,
        dropout: float = 0.0,
        rotary_pos_embs: Optional[nn.ModuleList] = None,
    ):
        """Initialize a transformer layer with attention and feed-forward components.

        Args:
            dim: Dimension of input/output tensors
            heads: Number of attention heads
            rotary_dim: Dimension for rotary embeddings, or None to disable
            rotary_base: Base for rotary embeddings
            dropout: Dropout probability for attention and feed-forward layers
            rotary_pos_embs: Optional pre-created rotary embeddings to share across layers
        """
        super().__init__()
        self.attn = Attention(
            dim=dim, heads=heads, dim_head=dim // heads, flash=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim, mult=2, dropout=dropout)

        if rotary_pos_embs is not None:
            self.rotary_pos_embs = rotary_pos_embs
        elif rotary_dim is not None:
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
            rotary_pos_emb = compute_rotary_emb(coords, self.rotary_pos_embs)

        context_rotary_pos_emb = None
        if context_coords is not None and self.rotary_pos_embs is not None:
            assert context_coords.shape[:2] == context.shape[:2]
            context_rotary_pos_emb = compute_rotary_emb(context_coords, self.rotary_pos_embs)

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


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 4,
        rotary_dim: Optional[int] = None,
        rotary_dim_context: Optional[int] = None,
        rotary_base: int = 10000,
        dropout: float = 0.0,
        ff_mult: int = 4,
        with_cross_attn: bool = True,
        rotary_pos_embs: Optional[nn.ModuleList] = None,
        rotary_pos_embs_context: Optional[nn.ModuleList] = None,
    ):
        """Transformer decoder layer with self-attention, cross-attention, and feed-forward.

        Args:
            dim: Feature dimension
            heads: Number of attention heads
            rotary_dim: RoPE dimensionality for queries (None to disable)
            rotary_dim_context: RoPE dimensionality for context keys (None to disable)
            rotary_base: Base for RoPE
            dropout: Dropout rate
            ff_mult: Feedforward expansion factor (default 4 to match classical transformers)
            with_cross_attn: Whether to include cross-attention layer
            rotary_pos_embs: Pre-created RoPE modules for queries (shared across layers)
            rotary_pos_embs_context: Pre-created RoPE modules for context (shared across layers)
        """
        super().__init__()

        self.with_cross_attn = with_cross_attn

        # Self-attention
        self.self_attn = Attention(
            dim=dim, heads=heads, dim_head=dim // heads, flash=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)

        # Cross-attention (optional)
        if with_cross_attn:
            self.cross_attn = Attention(
                dim=dim, heads=heads, dim_head=dim // heads, flash=True, dropout=dropout
            )
            self.norm2 = nn.LayerNorm(dim)
            self.norm3 = nn.LayerNorm(dim)
        else:
            self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        self.ff = FeedForward(dim, dim, mult=ff_mult, dropout=dropout)

        # Rotary embeddings for QUERIES
        if rotary_pos_embs is not None:
            self.rotary_pos_embs = rotary_pos_embs
        elif rotary_dim is not None:
            dims = _split_rope_dims(dim, heads, rotary_dim)
            self.rotary_pos_embs = nn.ModuleList(
                [RotaryEmbeddingTrainable(dim=d, base=rotary_base) for d in dims]
            )
        else:
            self.rotary_pos_embs = None

        # Rotary embeddings for CONTEXT KEYS
        if rotary_pos_embs_context is not None:
            self.rotary_pos_embs_context = rotary_pos_embs_context
        elif rotary_dim_context is not None:
            dims_context = _split_rope_dims(dim, heads, rotary_dim_context)
            self.rotary_pos_embs_context = nn.ModuleList(
                [RotaryEmbeddingTrainable(dim=d, base=rotary_base) for d in dims_context]
            )
        else:
            self.rotary_pos_embs_context = None

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        coords: Optional[Tensor] = None,
        context_coords: Optional[Tensor] = None,
        level_idx: Optional[Tensor] = None,
        context_level_idx: Optional[Tensor] = None,
        attention_mode: Literal["all", "causal", "same", "random"] = "all",
    ):
        """Apply decoder layer: self-attention -> cross-attention -> feed-forward.

        Args:
            x: Queries (B, N, D)
            context: Context for cross-attention (B, M, D), None to skip cross-attention
            coords: Coordinates for query RoPE (B, N, ndim)
            context_coords: Coordinates for context key RoPE (B, M, ndim)
            level_idx: Level indices for queries
            context_level_idx: Level indices for context
            attention_mode: Attention masking mode

        Returns:
            Transformed tensor of same shape as input
        """
        B, N, D = x.shape

        # === SELF-ATTENTION ===

        # Compute RoPE for queries
        query_rotary_emb = None
        if coords is not None and self.rotary_pos_embs is not None:
            assert coords.shape[:2] == x.shape[:2]
            query_rotary_emb = compute_rotary_emb(coords, self.rotary_pos_embs)

        # Self-attention mask
        self_attn_mask = None
        if level_idx is not None:
            if attention_mode == "all":
                self_attn_mask = None
            elif attention_mode == "causal":
                self_attn_mask = level_idx.view(B, 1, N, 1) >= level_idx.view(B, 1, 1, N)
            elif attention_mode == "same":
                self_attn_mask = level_idx.view(B, 1, N, 1) == level_idx.view(B, 1, 1, N)
            elif attention_mode == "random":
                self_attn_mask = torch.argsort(torch.rand(N, N, device=x.device), dim=1) == 0
            else:
                raise ValueError(f"Invalid attention mode: {attention_mode}")

        # Apply self-attention
        x = x + self.self_attn(
            self.norm1(x),
            attn_mask=self_attn_mask,
            rotary_pos_emb=query_rotary_emb,
        )

        # === CROSS-ATTENTION (if enabled and context provided) ===

        if self.with_cross_attn and context is not None:
            M = context.shape[1]
            assert context.shape[0] == B and context.shape[2] == D

            # Compute RoPE for context keys
            context_rotary_emb = None
            if context_coords is not None and self.rotary_pos_embs_context is not None:
                assert context_coords.shape[:2] == context.shape[:2]
                context_rotary_emb = compute_rotary_emb(
                    context_coords, self.rotary_pos_embs_context
                )

            # Cross-attention mask
            cross_attn_mask = None
            if level_idx is not None and context_level_idx is not None:
                if attention_mode == "all":
                    cross_attn_mask = None
                elif attention_mode == "causal":
                    cross_attn_mask = level_idx.view(B, 1, N, 1) >= context_level_idx.view(B, 1, 1, M)
                elif attention_mode == "same":
                    cross_attn_mask = level_idx.view(B, 1, N, 1) == context_level_idx.view(B, 1, 1, M)

            # Apply cross-attention
            x = x + self.cross_attn(
                self.norm2(x),
                context=context,
                attn_mask=cross_attn_mask,
                rotary_pos_emb=None,  # Queries don't get RoPE in cross-attention
                context_rotary_pos_emb=context_rotary_emb,
            )

            x = x + self.ff(self.norm3(x))
        else:
            # No cross-attention, just feed-forward
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
        if hasattr(cls, "ndim") and "ndim" in config.keys():
            if config["ndim"] != cls.ndim:
                raise ValueError(f"Expected ndim={cls.ndim}, but got {config['ndim']}. Are you sure you're instantiating the correct class?")
        if "ndim" in config:
            config.pop("ndim", None)

        # Backward compatibility: convert old use_rotary_embed to rotary_mode
        if "use_rotary_embed" in config:
            if "rotary_mode" in config:
                raise ValueError(
                    "Config contains both 'use_rotary_embed' (deprecated) and 'rotary_mode'. "
                    "Please use only 'rotary_mode'."
                )
            config["rotary_mode"] = "per_layer" if config["use_rotary_embed"] else "none"
            config.pop("use_rotary_embed")

        model = cls(**config).to(device)
        model.load_state_dict(torch.load(path / "model.pth", map_location=device))
        return model
