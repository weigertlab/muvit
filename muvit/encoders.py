from abc import ABC, abstractmethod
from typing import Generic, Literal, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
import torch_dct
from einops import rearrange
from torch import Tensor, nn
from x_transformers.x_transformers import RotaryEmbedding

from .bblocks import SaveableModel, TransformerLayer, RotaryEmbeddingTrainable

T = TypeVar("T", bound=Tuple[int, ...])

def patch_dct(x:Tensor, in_channels:int, patch_size:tuple[int,int], inverse:bool=False, norm:Literal['ortho']='ortho') -> Tensor:
    """Apply DCT to each patch of a 2D image.
    
    Args:
        x: Input patches of shape (B, N, P*P*C)
        in_channels: Number of input channels
        patch_size: Size of image patches
        inverse: Whether to apply inverse DCT
    
    Returns:
        Tensor of shape (B, N, P*P*C)
    """
    if len(patch_size) == 2:
        func = torch_dct.dct_2d if not inverse else torch_dct.idct_2d
        x = rearrange(x, 'b n (p1 p2 c) -> b (n c) p1 p2', p1=patch_size[0], p2=patch_size[1], c=in_channels)
        x = func(x, norm=norm)
        x = rearrange(x, 'b (n c) p1 p2 -> b n (p1 p2 c)', c=in_channels)
    elif len(patch_size) == 3:
        func = torch_dct.dct_3d if not inverse else torch_dct.idct_3d
        x = rearrange(x, 'b n (p1 p2 p3 c) -> b (n c) p1 p2 p3', p1=patch_size[0], p2=patch_size[1], p3=patch_size[2], c=in_channels)
        x = func(x, norm=norm)
        x = rearrange(x, 'b (n c) p1 p2 p3 -> b n (p1 p2 p3 c)', c=in_channels)
    else:
        raise ValueError(f"Invalid patch size: {patch_size} (must be 2 or 3)")
    
    return x

class MuViTEncoder(SaveableModel, ABC, Generic[T]):
    def __init__(
        self,
        in_channels: int,
        levels: tuple[float],
        patch_size: Union[int, tuple[int, ...]],
        num_layers: int = 6,
        dim: int = 320,
        heads: int = 4,
        attention_mode: Literal["all", "causal", "same", "random"] = "all",
        use_level_embed: bool = True,
        rotary_mode: Literal["none", "fixed", "shared", "per_layer"] = "per_layer",
        rotary_base: int = 10000,
        input_space: Literal["real", "dct"] = "real",
        dropout: float = 0.0,
        use_rotary_embed: Optional[bool] = None,
    ):
        """Initialize a multi-level Vision Transformer encoder.

        Args:
            in_channels: Number of input channels
            levels: Tuple of scale factors for each level
            patch_size: Size of image patches
            num_layers: Number of transformer layers
            dim: Hidden dimension
            heads: Number of attention heads
            attention_mode: Type of attention masking
            use_level_embed: Whether to use level embeddings
            rotary_mode: Type of rotary embeddings (none/fixed/shared/per_layer)
            rotary_base: Base for rotary embeddings
            input_space: Input space (real/dct)
            dropout: Dropout probability
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

        self.levels = levels
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size,) * self.ndim
        )
        self.dim = dim
        self.attention_mode = attention_mode
        self.in_channels = in_channels

        self._config = dict(
            in_channels=in_channels,
            levels=levels,
            patch_size=patch_size,
            num_layers=num_layers,
            dim=dim,
            heads=heads,
            attention_mode=attention_mode,
            use_level_embed=use_level_embed,
            rotary_mode=rotary_mode,
            rotary_base=rotary_base,
            dropout=dropout,
            input_space=input_space,
            ndim=self.ndim,
        )

        self.input_space = input_space
        patch_dim = in_channels * np.prod(self.patch_size)
        self.proj = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(patch_dim, dim), nn.LayerNorm(dim))
                for _ in range(len(levels))
            ]
        )

        if use_level_embed:
            self.level_embed = nn.Parameter(torch.randn(len(levels), 1, 1, dim))
        else:
            self.level_embed = None

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

    @classmethod
    @property
    @abstractmethod
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        pass

    def patchify(self, x: Tensor) -> Tensor:
        """Convert input tensor to patches.

        Input: (B,C,(T),(D),H,W) -> Output: (B,N,(P*)(P*)P*P*C)
        """
        if self.ndim == 2:
            # B, C, H, W
            x = rearrange(
                x,
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
        elif self.ndim == 3:
            # B, C, D, H, W
            x = rearrange(
                x,
                "b c (d p1) (h p2) (w p3) -> b (d h w) (p1 p2 p3 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
        elif self.ndim == 4:
            # B, C, T, D, H, W
            x = rearrange(
                x,
                "b c (t p1) (d p2) (h p3) (w p4) -> b (t d h w) (p1 p2 p3 p4 c)",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
                p4=self.patch_size[3],
            )
        else:
            raise ValueError(f"Invalid number of dimensions: {self.ndim}")
        return x

    @abstractmethod
    def get_patch_coords(
        self, B: int, L: int, *shape: int, bbox: Optional[tuple[Tensor, Tensor]] = None
    ) -> Tensor:
        """Get coordinates for each patch."""
        pass

    @abstractmethod
    def patch_embed(
        self, x: Tensor, bbox: Optional[tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process multi-level input through patch embedding."""
        pass

    def patch_space_transform(self, patches: Tensor, inverse: bool = False) -> Tensor:
        """transform real or dct space."""
        if self.input_space == "real":
            return patches
        elif self.input_space == "dct":
            B, N, _ = patches.shape
            return patch_dct(
                patches.float(), self.in_channels, self.patch_size, inverse=inverse
            ).to(patches.dtype)
        else:
            raise ValueError(f"Invalid input space: {self.input_space}")

    def forward_masked(
        self,
        x: Tensor,
        bbox: Optional[Tensor] = None,
        masking_ratio: float = 0.75,
        masking_mode: Literal["dirichlet", "random"] | tuple[float] = "dirichlet",
    ):
        x, patches, coords = self.patch_embed(x, bbox)

        B, N, D = x.shape
        if not N % len(self.levels) == 0:
            raise ValueError(
                f"Number of patches must be divisible by number of levels: {N} % {len(self.levels)} != 0"
            )

        N_per_level = N // len(self.levels)
        N_retained = int(N * (1 - masking_ratio))

        level_idx = (
            torch.arange(patches.shape[1], device=x.device)
            .unsqueeze(0)
            .repeat(patches.shape[0], 1)
        )
        level_idx = level_idx // N_per_level

        if masking_mode == "dirichlet":
            # alpha_parameter = 1.0
            alpha_parameter = 0.5
            prob_weights = torch.distributions.Dirichlet(
                torch.ones(len(self.levels), device=x.device) * alpha_parameter
            ).sample()
        elif masking_mode == "random":
            prob_weights = torch.ones(len(self.levels), device=x.device)
        elif isinstance(masking_mode, (tuple, list, np.ndarray)):
            if not len(masking_mode) == len(self.levels):
                raise ValueError(
                    f"Number of levels must match number of masking probabilities: {len(masking_mode)} != {len(self.levels)}"
                )
            prob_weights = torch.tensor(masking_mode, device=x.device)
        else:
            raise ValueError(f"Invalid masking mode: {masking_mode}")

        prob_per_block = torch.repeat_interleave(prob_weights, N_per_level)
        idx = torch.stack(
            [torch.multinomial(prob_per_block, N, replacement=False) for _ in range(B)],
            dim=0,
        )

        idx_retain, idx_mask = idx[:, :N_retained], idx[:, N_retained:]

        batch_range = torch.arange(B, device=x.device)[:, None]
        x = x[batch_range, idx_retain]
        level_idx = level_idx[batch_range, idx_retain]

        for layer in self.layers:
            x = layer(
                x,
                level_idx=level_idx,
                coords=coords[batch_range, idx_retain],
                attention_mode=self.attention_mode,
            )

        return x, coords, patches, batch_range, idx_retain, idx_mask

    def forward(self, x: Tensor, bbox: Optional[Tensor] = None, return_intermediate_idxs: Optional[Tuple[int]] = None):
        """Process multi-level input through the encoder."""
        x, patches, coords = self.patch_embed(x, bbox)
        B, N, D = x.shape

        N_per_level = N // len(self.levels)

        level_idx = (
            torch.arange(patches.shape[1], device=x.device)
            .unsqueeze(0)
            .repeat(patches.shape[0], 1)
        )
        level_idx = level_idx // N_per_level
        if return_intermediate_idxs is not None:
            interm_outputs = []
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                level_idx=level_idx,
                coords=coords,
                attention_mode=self.attention_mode,
            )
            if return_intermediate_idxs is not None and i in return_intermediate_idxs:
                interm_outputs.append(x)

        if return_intermediate_idxs is not None:
            return x, coords, level_idx, tuple(interm_outputs)
        else:
            return x, coords, level_idx


class MuViTEncoder2d(MuViTEncoder[Tuple[int, int]]):
    @classmethod
    @property
    def ndim(self) -> int: # FIXME: breaks in py313 as classmethod+properties are deprecated
        return 2

    def get_patch_coords(
        self,
        B: int,
        L: int,
        H: int,
        W: int,
        bbox: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Get coordinates for each patch.

        Args:
            B: Batch size
            L: Number of levels
            H: Height of input
            W: Width of input
            bbox: Optional bounding box coordinates defining the spatial extent of each crop.
                 Shape (B, L, 2, 2) where dims are (batch, level, top/bottom, y/x).
        Returns: (B,L*H*W,2) coordinates
        """
        h, w = H // self.patch_size[0], W // self.patch_size[1]

        if bbox is None:
            bbox = (
                [[-l * (H / 2 - 0.5), -l * (W / 2 - 0.5)] for l in self.levels],
                [[l * (H / 2 - 0.5), l * (W / 2 - 0.5)] for l in self.levels],
            )
            bbox = torch.tensor(bbox).permute(1, 0, 2)
            bbox = bbox.view(1, L, 2, 2).repeat(B, 1, 1, 1)

        if not bbox.ndim == 4 and bbox.shape[:3] == (B, L, 2):
            raise ValueError(
                f"Bounding box must be of shape (B, L, 2, 2): {bbox.shape}"
            )

        coords = torch.meshgrid(
            torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij"
        )
        coords = torch.stack(coords, dim=-1).to(bbox.device)  # (h, w, 2)

        coords = coords.view(1, 1, h, w, 2)  # (1, 1, h, w, 2)
        coords = coords.repeat(B, L, 1, 1, 1)  # (B, L, h, w, 2)

        # Scale and offset coords according to bbox
        top_left = bbox[:, :, 0, :].view(B, L, 1, 1, 2)  # (B, L, 1, 1, 2)
        bottom_right = bbox[:, :, 1, :].view(B, L, 1, 1, 2)  # (B, L, 1, 1, 2)
        coords = top_left + coords * (bottom_right - top_left)  # (B, L, h, w, 2)

        coords = rearrange(coords, "b l h w c -> b (l h w) c")
        return coords.to(bbox.device)

    def patch_embed(
        self, x: Tensor, bbox: Optional[tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process multi-level input through patch embedding.

        Input: x (B, L, C, H, W), bbox (B, L, 2, 2)
        Output: x (B, N, D), patches (B, N, C*P*P), coords (B, N, 2)
        """
        B, L, C, H, W = x.shape
        assert L == len(self.levels), "Number of levels must match"

        if bbox is not None:
            expected_shape = (B, L, 2, 2)
            if not (bbox.ndim == 4 and bbox.shape[:3] == (B, L, 2)):
                raise ValueError(
                    f"Bounding box must be of shape {expected_shape}: {bbox.shape}"
                )

        coords = self.get_patch_coords(B, L, H, W, bbox=bbox).to(x.device)

        # patchify assumes batch dimension is first, so we loop over levels
        x = x.transpose(0, 1)
        patches = tuple(self.patch_space_transform(self.patchify(_x)) for _x in x)
        x = tuple(pro(pat) for pro, pat in zip(self.proj, patches))

        if self.level_embed is not None:
            x = tuple(x + le for x, le in zip(x, self.level_embed))

        # back to B, L order
        x = torch.cat(x, dim=1)
        patches = torch.cat(patches, dim=1)

        return x, patches, coords

    def compute_features(self, x: Tensor, bbox: Optional[Tensor] = None) -> Tensor:
        """
        Compute features from a given input stack/bbox and return them in a spatially structured way.

        Args:
            x: Input tensor of shape (B, L, C, H, W)
            bbox: Optional bounding box coordinates of shape (B, L, 2, 2)

        Returns:
            feats: Features of shape (B, L, D, H', W') where H' = H // patch_size[0], W' = W // patch_size[1], D is the feature dimension
        """
        feats, _, _ = self(x, bbox=bbox)
        H, W = x.shape[-2], x.shape[-1]
        L = len(self.levels)
        h = H // self.patch_size[0]
        w = W // self.patch_size[1]
        return rearrange(feats, "b (l h w) d -> b l d h w", l=L, h=h, w=w)


class MuViTEncoder3d(MuViTEncoder[Tuple[int, int, int]]):
    @classmethod
    @property
    def ndim(self) -> int:
        return 3

    def get_patch_coords(
        self,
        B: int,
        L: int,
        D: int,
        H: int,
        W: int,
        bbox: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Get coordinates for each patch.

        Returns: (B,L*D*H*W,3) coordinates
        """
        d, h, w = (
            D // self.patch_size[0],
            H // self.patch_size[1],
            W // self.patch_size[2],
        )

        if bbox is None:
            bbox = (
                [
                    [-l * (D / 2 - 0.5), -l * (H / 2 - 0.5), -l * (W / 2 - 0.5)]
                    for l in self.levels
                ],
                [
                    [l * (D / 2 - 0.5), l * (H / 2 - 0.5), l * (W / 2 - 0.5)]
                    for l in self.levels
                ],
            )
            bbox = torch.tensor(bbox).permute(1, 0, 2)
            bbox = bbox.view(1, L, 2, 3).repeat(B, 1, 1, 1)

        if not bbox.ndim == 4 and bbox.shape[:3] == (B, L, 2):
            raise ValueError(
                f"Bounding box must be of shape (B, L, 2, 3): {bbox.shape}"
            )

        coords = torch.meshgrid(
            torch.linspace(0, 1, d),
            torch.linspace(0, 1, h),
            torch.linspace(0, 1, w),
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).to(bbox.device)  # (d, h, w, 3)
        coords = coords.view(1, 1, d, h, w, 3)  # (1, 1, d, h, w, 3)
        coords = coords.repeat(B, L, 1, 1, 1, 1)  # (B, L, d, h, w, 3)

        # Scale and offset coords according to bbox
        top_left = bbox[:, :, 0, :].view(B, L, 1, 1, 1, 3)  # (B, L, 1, 1, 1, 3)
        bottom_right = bbox[:, :, 1, :].view(B, L, 1, 1, 1, 3)  # (B, L, 1, 1, 1, 3)
        coords = top_left + coords * (bottom_right - top_left)  # (B, L, d, h, w, 3)

        coords = rearrange(coords, "b l d h w c -> b (l d h w) c")
        return coords.to(bbox.device)

    def patch_embed(
        self, x: Tensor, bbox: Optional[tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process multi-level input through patch embedding.

        Input: x (B, L, C, D, H, W), bbox (B, L, 2, 3)
        Output: x (B, N, D), patches (B, N, C*D*H*W), coords (B, N, 3)
        """
        B, L, C, D, H, W = x.shape
        assert L == len(self.levels), "Number of levels must match"

        if bbox is not None:
            expected_shape = (B, L, 2, 3)
            if not (bbox.ndim == 4 and bbox.shape[:3] == (B, L, 2)):
                raise ValueError(
                    f"Bounding box must be of shape {expected_shape}: {bbox.shape}"
                )

        coords = self.get_patch_coords(B, L, D, H, W, bbox=bbox).to(x.device)

        # patchify assumes batch dimension is first, so we loop over levels
        x = x.transpose(0, 1)
        patches = tuple(self.patch_space_transform(self.patchify(_x)) for _x in x)
        x = tuple(pro(pat) for pro, pat in zip(self.proj, patches))

        if self.level_embed is not None:
            x = tuple(x + le for x, le in zip(x, self.level_embed))

        # back to B, L order
        x = torch.cat(x, dim=1)
        patches = torch.cat(patches, dim=1)

        return x, patches, coords

    def compute_features(self, x: Tensor, bbox: Optional[Tensor] = None) -> Tensor:
        """
        Compute features from a given input stack/bbox and return them in a spatially structured way.

        Args:
            x: Input tensor of shape (B, L, C, D, H, W)
            bbox: Optional bounding box coords

        Returns:
            feats: Features of shape (B, L, N, D', H', W') where H' = H // patch_size[0], W' = W // patch_size[1], D is the feature dimension
        """
        feats, _, _ = self(x, bbox=bbox)
        D, H, W = x.shape[-3], x.shape[-2], x.shape[-1]
        L = len(self.levels)
        d = D // self.patch_size[0]
        h = H // self.patch_size[1]
        w = W // self.patch_size[2]
        return rearrange(feats, "b (l d h w) n -> b l n d h w", l=L, h=h, w=w, d=d)


class MuViTEncoder4d(MuViTEncoder[Tuple[int, int, int]]):
    @classmethod
    @property
    def ndim(self) -> int:
        return 4

    def get_patch_coords(
        self,
        B: int,
        L: int,
        T: int,
        D: int,
        H: int,
        W: int,
        bbox: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Get coordinates for each patch.

        Returns: (B,L*D*H*W,3) coordinates
        """
        t, d, h, w = (
            T // self.patch_size[0],
            D // self.patch_size[1],
            H // self.patch_size[2],
            W // self.patch_size[3],
        )

        if bbox is None:
            bbox = (
                [
                    [-lv * (T / 2 - 0.5), -lv * (D / 2 - 0.5), -lv * (H / 2 - 0.5), -lv * (W / 2 - 0.5)]
                    for lv in self.levels
                ],
                [
                    [lv * (T / 2 - 0.5), lv * (D / 2 - 0.5), lv * (H / 2 - 0.5), lv * (W / 2 - 0.5)]
                    for lv in self.levels
                ],
            )
            bbox = torch.tensor(bbox).permute(1, 0, 2)
            bbox = bbox.view(1, L, 2, self.ndim).repeat(B, 1, 1, 1)

        if not bbox.ndim == 4 and bbox.shape[:3] == (B, L, 4):
            raise ValueError(
                f"Bounding box must be of shape (B, L, 2, 4): {bbox.shape}"
            )
        coords = torch.meshgrid(
            torch.linspace(0, 1, t),
            torch.linspace(0, 1, d),
            torch.linspace(0, 1, h),
            torch.linspace(0, 1, w),
            indexing="ij",
        )
        coords = torch.stack(coords, dim=-1).to(bbox.device)  # (t, d, h, w, 4)
        coords = coords.view(1, 1, t, d, h, w, 4)  # (1, 1, t, d, h, w, 4)
        coords = coords.repeat(B, L, 1, 1, 1, 1, 1)  # (B, L, t, d, h, w, 4)

        # Scale and offset coords according to bbox
        top_left = bbox[:, :, 0, :].view(B, L, 1, 1, 1, 1, self.ndim)  # (B, L, 1, 1, 1, 1, 3)
        bottom_right = bbox[:, :, 1, :].view(B, L, 1, 1, 1, 1, self.ndim)  # (B, L, 1, 1, 1, 1, 3)
        coords = top_left + coords * (bottom_right - top_left)  # (B, L, t, d, h, w, 3)

        coords = rearrange(coords, "b l t d h w c -> b (l t d h w) c")
        return coords.to(bbox.device)

    def patch_embed(
        self, x: Tensor, bbox: Optional[tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Process multi-level input through patch embedding.

        Input: x (B, L, C, T, D, H, W), bbox (B, L, 2, 4)
        Output: x (B, N, D), patches (B, N, C*T*D*H*W), coords (B, N, 4)
        """
        B, L, C, T, D, H, W = x.shape
        assert L == len(self.levels), "Number of levels must match"

        if bbox is not None:
            expected_shape = (B, L, 2, 4)
            if not (bbox.ndim == 4 and bbox.shape[:3] == (B, L, 2)):
                raise ValueError(
                    f"Bounding box must be of shape {expected_shape}: {bbox.shape}"
                )

        coords = self.get_patch_coords(B, L, T, D, H, W, bbox=bbox).to(x.device)

        # patchify assumes batch dimension is first, so we loop over levels
        x = x.transpose(0, 1)
        patches = tuple(self.patch_space_transform(self.patchify(_x)) for _x in x)
        x = tuple(pro(pat) for pro, pat in zip(self.proj, patches))

        if self.level_embed is not None:
            x = tuple(x + le for x, le in zip(x, self.level_embed))

        # back to B, L order
        x = torch.cat(x, dim=1)
        patches = torch.cat(patches, dim=1)

        return x, patches, coords

    def compute_features(self, x: Tensor, bbox: Optional[Tensor] = None) -> Tensor:
        """
        Compute features from a given input stack/bbox and return them in a spatially structured way.

        Args:
            x: Input tensor of shape (B, L, C, T, D, H, W)
            bbox: Optional bounding box coords

        Returns:
            feats: Features of shape (B, L, N, T', D', H', W') where H' = H // patch_size[0], W' = W // patch_size[1], D is the feature dimension
        """
        feats, _, _ = self(x, bbox=bbox)
        T, D, H, W = x.shape[-4], x.shape[-3], x.shape[-2], x.shape[-1]
        L = len(self.levels)
        t = T // self.patch_size[0]
        d = D // self.patch_size[1]
        h = H // self.patch_size[2]
        w = W // self.patch_size[3]
        return rearrange(feats, "b (l t d h w) n -> b l n t d h w", l=L, t=t, d=d, h=h, w=w)