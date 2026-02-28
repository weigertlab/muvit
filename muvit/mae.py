import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, Optional, Tuple, Type, TypeVar, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from argparse import Namespace
from einops import rearrange
from torch import Tensor, nn

from .bblocks import SaveableModel
from .decoders import MuViTDecoder, MuViTDecoder2d, MuViTDecoder3d, MuViTDecoder4d
from .encoders import MuViTEncoder, MuViTEncoder2d, MuViTEncoder3d, MuViTEncoder4d
from .trainer import WrappedModel

T = TypeVar("T", bound=Tuple[int, ...])


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
console_handler.setFormatter(formatter)
log.addHandler(console_handler)


class MuViTMAE(SaveableModel, ABC, Generic[T]):
    def __init__(
        self,
        in_channels: int = 1,
        levels: tuple[float] = (1, 4),
        patch_size: Union[int, tuple[int, ...]] = 16,
        num_layers: int = 12,
        dim: int = 512,
        num_layers_decoder: int = 2,
        dim_decoder: Optional[int] = 256,
        heads: int = 8,
        decoder_mode: Literal["single", "multi", "multi_iso"] = "multi",
        loss: Literal["mse", "norm_mse"] = "mse",
        masking_ratio: float = 0.75,
        use_level_embed: bool = True,
        rotary_mode: Literal["none", "fixed", "shared", "per_layer"] = "per_layer",
        rotary_base: int = 10000,
        attention_mode: Literal["all", "causal", "same", "random"] = "all",
        masking_mode: Literal["dirichlet", "random"] | tuple[float] = "dirichlet",
        input_space: Literal["real"] = "real",
        dropout: float = 0.0,
    ):
        """Initialize a Masked Autoencoder with multi-level Vision Transformer.

        Args:
            in_channels: Number of input channels
            levels: Tuple of scale factors for each level
            patch_size: Size of image patches
            num_layers: Number of encoder layers
            dim: Hidden dimension
            num_layers_decoder: Number of decoder layers
            dim_decoder: Hidden dimension for decoder
            heads: Number of attention heads
            decoder_mode: Type of decoder ('single', 'multi', 'multi_iso')
            loss: Type of loss function ('mse', 'norm_mse')
            masking_ratio: Ratio of patches to mask
            use_level_embed: Whether to use level embeddings
            rotary_mode: Type of rotary embeddings (none/fixed/shared/per_layer)
            rotary_base: Base for rotary embeddings
            attention_mode: Type of attention masking
            masking_mode: Type of masking ('dirichlet', 'random' or tuple of probabilities)
            dropout: Dropout probability for transformer layers
        """
        super().__init__()

        if dim_decoder is None:
            dim_decoder = dim

        # Select appropriate encoder/decoder based on spatial dimensionality
        EncoderCls = self.encoder_class
        DecoderCls = self.decoder_class

        self._config = dict(
            in_channels=in_channels,
            levels=levels,
            patch_size=patch_size,
            num_layers=num_layers,
            dim=dim,
            heads=heads,
            use_level_embed=use_level_embed,
            rotary_mode=rotary_mode,
            rotary_base=rotary_base,
            decoder_mode=decoder_mode,
            loss=loss,
            masking_ratio=masking_ratio,
            masking_mode=masking_mode,
            attention_mode=attention_mode,
            num_layers_decoder=num_layers_decoder,
            dim_decoder=dim_decoder,
            dropout=dropout,
            input_space=input_space,
            ndim=self.ndim,
        )

        self.encoder = EncoderCls(
            in_channels=in_channels,
            levels=levels,
            patch_size=patch_size,
            num_layers=num_layers,
            dim=dim,
            heads=heads,
            use_level_embed=use_level_embed,
            rotary_mode=rotary_mode,
            rotary_base=rotary_base,
            attention_mode=attention_mode,
            dropout=dropout,
            input_space=input_space,
        )

        # Calculate patch dimension for final layer
        patch_dim = in_channels * np.prod(self.encoder.patch_size)
        self.final = nn.Linear(dim_decoder, patch_dim)

        self.patch_size = self.encoder.patch_size
        self.masking_ratio = masking_ratio
        self.masking_mode = masking_mode
        self.decoder_mode = decoder_mode

        if decoder_mode == "single":
            self.mask_token = nn.Parameter(torch.randn(1, 1, dim))
            self.decoder = DecoderCls(
                dim,
                dim_decoder,
                num_layers=num_layers_decoder,
                heads=heads,
                rotary_mode=rotary_mode,
                rotary_base=rotary_base,
                dropout=dropout,
            )
        elif decoder_mode in ("multi", "multi_iso"):
            self.mask_token = nn.Parameter(
                torch.randn(1, len(self.encoder.levels), dim)
            )
            self.decoder = nn.ModuleList(
                [
                    DecoderCls(
                        dim,
                        dim_decoder,
                        num_layers=num_layers_decoder,
                        heads=heads,
                        rotary_mode=rotary_mode,
                        rotary_base=rotary_base,
                        dropout=dropout,
                    )
                    for _ in range(len(self.encoder.levels))
                ]
            )
        else:
            raise ValueError(f"Invalid decoder mode: {decoder_mode}")
        self.loss_fn = loss

    @classmethod
    @property
    @abstractmethod
    def encoder_class(self) -> Type[MuViTEncoder]:
        """Get the appropriate encoder class for this dimensionality."""
        pass

    @classmethod
    @property
    @abstractmethod
    def decoder_class(self) -> Type[MuViTDecoder]:
        """Get the appropriate decoder class for this dimensionality."""
        pass

    @classmethod
    @property
    @abstractmethod
    def ndim(self) -> int:
        """Get the number of dimensions for this model."""
        pass

    @abstractmethod
    def patch_token_to_image(
        self, x: Tensor, *shape: int, space_transform: bool = True
    ) -> Tensor:
        """Convert patch tokens back to image."""
        pass

    def token_to_patch(self, x: Tensor) -> Tensor:
        if self.ndim == 2:
            return rearrange(
                x,
                "b n (p1 p2) -> b n p1 p2",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
            )
        elif self.ndim == 3:
            return rearrange(
                x,
                "b n (p1 p2 p3) -> b n p1 p2 p3",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
            )
        elif self.ndim == 4:
            return rearrange(
                x,
                "b n (p1 p2 p3 p4) -> b n p1 p2 p3 p4",
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2],
                p4=self.patch_size[3],
            )
        else:
            raise ValueError(f"Invalid number of dimensions: {self.ndim}")

    def forward(
        self,
        x: Tensor,
        bbox: Optional[tuple[Tensor, Tensor]] = None,
        return_all: bool = False,
        eps: float = 1e-2,
        consistent_levels: bool=False,
        rng_generator: Optional[torch.Generator]=None,
        masking_mode_is_ratio: bool=False,
    ):
        """Process input through the MAE model.

        Args:
            x: Input tensor
            bbox: Optional bounding box coordinates
            return_all: Whether to return intermediate results
            eps: Small constant for numerical stability
            consistent_levels: Whether to use consistent masking levels across the batch. If True, the same token locations will be masked out for different runs (assuming same RNG), otherwise different runs may have different masked tokens. Defaults to False.
            rng_generator: Optional random number generator for reproducibility of the `consistent_levels` option.
            masking_mode_is_ratio: Whether the masking_mode is given as a ratio (True) denoting the explicit fraction of tokens to mask, or as a tuple of probabilities (False).

        Returns:
            Dictionary containing:
            - input: Original patches
            - output: Reconstructed patches
            - loss: Reconstruction loss
            - encoded: Encoded features (if return_all)
            - reco: Reconstructed image (if return_all)
            - decoded: Decoded features (if return_all)
            - input_masked: Masked input (if return_all)
            - loss_per_level: Loss per level (if return_all)
        """
        y, coords, patches, batch_range, idx_retain, idx_mask = (
            self.encoder.forward_masked(x, bbox, self.masking_ratio, self.masking_mode, consistent_levels, rng_generator, masking_mode_is_ratio=masking_mode_is_ratio)
        )

        N = patches.shape[1]
        N_per_level = N // len(self.encoder.levels)

        z = torch.zeros(
            patches.shape[0], N, y.shape[-1], device=patches.device, dtype=patches.dtype
        )

        if self.decoder_mode == "single":
            z[batch_range, idx_mask] = self.mask_token
            z[batch_range, idx_retain] = y
            # single self-attention for all levels
            z = self.decoder(z, coords)
        elif self.decoder_mode in ("multi", "multi_iso"):
            mask_tokens = torch.repeat_interleave(
                self.mask_token, N_per_level, dim=1
            ).repeat(patches.shape[0], 1, 1)
            z[batch_range, idx_mask] = mask_tokens[batch_range, idx_mask]
            z[batch_range, idx_retain] = y
            zs = torch.split(z, N_per_level, dim=1)
            cs = torch.split(coords, N_per_level, dim=1)
            if self.decoder_mode == "multi":
                # allow cross attending to all levels
                z = torch.cat(
                    [
                        self.decoder[i](_z, _c, context=z, context_coords=coords)
                        for i, (_z, _c) in enumerate(zip(zs, cs))
                    ],
                    dim=1,
                )
            elif self.decoder_mode == "multi_iso":
                # do it in isolation for each level
                z = torch.cat(
                    [self.decoder[i](_z, _c) for i, (_z, _c) in enumerate(zip(zs, cs))],
                    dim=1,
                )
            else:
                raise ValueError(f"Invalid decoder mode: {self.decoder_mode}")
        else:
            raise ValueError(f"Invalid decoder mode: {self.decoder_mode}")

        z = self.final(z)

        # normalize target
        if self.loss_fn == "norm_mse":
            p_mean = patches.mean(dim=-1, keepdim=True)
            p_std = patches.std(dim=-1, keepdim=True)
        elif self.loss_fn in ("mse", "mse_fft"):
            p_mean = torch.tensor(0, device=x.device, dtype=patches.dtype)
            p_std = torch.tensor(1 - eps, device=x.device, dtype=patches.dtype)
        else:
            raise ValueError(f"Invalid loss: {self.loss_fn}")

        patches_normed = (patches - p_mean) / (p_std + eps)

        loss = F.mse_loss(
            z[batch_range, idx_mask], patches_normed[batch_range, idx_mask]
        )
        if self.loss_fn == "mse_fft":
            z2 = self.token_to_patch(z[batch_range, idx_mask]).to(torch.float32)
            patches2 = self.token_to_patch(patches_normed[batch_range, idx_mask]).to(
                torch.float32
            )
            z2f = torch.fft.rfftn(z2, dim=tuple(-(i + 1) for i in range(self.ndim)))
            patches2f = torch.fft.rfftn(
                patches2, dim=tuple(-(i + 1) for i in range(self.ndim))
            )
            loss = loss + 0.01 * F.l1_loss(z2f, patches2f)

        out = dict(input=patches, output=z, loss=loss)
        if return_all:
            reco = patches_normed.clone().to(z.dtype)
            reco[batch_range, idx_mask] = z[batch_range, idx_mask]
            reco = reco * (p_std + eps) + p_mean
            reco = self.encoder.patch_space_transform(reco, inverse=True)
            reco = self.patch_token_to_image(reco, *x.shape[2:]).to(patches.dtype)

            input_masked = patches.clone()
            input_masked = self.encoder.patch_space_transform(
                input_masked, inverse=True
            )
            input_masked[batch_range, idx_mask] = 0.5
            input_masked = self.patch_token_to_image(input_masked, *x.shape[2:])
            mask = torch.zeros(patches.shape, dtype=bool, device=x.device)
            mask[batch_range, idx_mask] = True
            mask = self.patch_token_to_image(mask, *x.shape[2:])
            
            # loss per lv
            B = idx_mask.shape[0]
            loss_per_level_list = []
            for i in range(len(self.encoder.levels)):
                sel = (idx_mask // N_per_level) == i  # (B, M_mask)

                selected_z = []
                selected_p = []
                for b in range(B):
                    sel_b = sel[b]
                    if sel_b.any():
                        idxs = idx_mask[b, sel_b]
                        selected_z.append(z[b, idxs])
                        selected_p.append(patches_normed[b, idxs])

                if len(selected_z) == 0:
                    loss_per_level_list.append(torch.tensor(0.0, device=x.device))
                else:
                    zz = torch.cat(selected_z, dim=0)
                    pp = torch.cat(selected_p, dim=0)
                    loss_per_level_list.append(F.mse_loss(zz, pp))

            loss_per_level = torch.stack(loss_per_level_list).to(x.device)

            out["encoded"] = y
            out["reco"] = reco
            out["decoded"] = z
            out["mask"] = mask
            out["input_masked"] = input_masked
            out["loss_per_level"] = loss_per_level
        return out

    def fit(
        self,
        train_dataloader,
        val_dataloader,
        output: Union[Path, str, None],
        num_epochs: int = 500,
        lr: Optional[float] = None,
        nobox: bool = False,
        warmup_epochs: int = 2,
        logger: Optional[str] = "wandb",
        run_name: Optional[str] = None,
        wandb_project: Optional[str] = "muvit",
        ckpt_path: Optional[Union[str, Path]] = None,
        accelerator: Optional[str] = "auto",
        num_nodes: Optional[int] = 1,
        precision: Optional[str] = "bf16-mixed",
        strategy: Optional[str] = "auto",
        data_gen: Optional[torch.Generator] = None,
        dry: Optional[bool] = False,
        gradient_clip_val: Optional[float] = 0.5,
        fast_dev_run: bool = False,
        args_namespace: Optional[Namespace] = None
    ):
        if num_nodes < 1:
            raise ValueError("The number of nodes should be at least 1.")

        if logger in ("tensorboard", "tb"):
            logger = pl.loggers.TensorBoardLogger(save_dir=output, name=run_name)
        elif logger == "wandb":
            logger = pl.loggers.WandbLogger(
                project=wandb_project, name=run_name, save_dir=output
            )
            if args_namespace is not None:
                logger.log_hyperparams(vars(args_namespace))
        elif logger is None:
            if not dry:
                log.warning(
                    "No logger provided. Training will run, but progress will not be logged."
                )
        else:
            raise ValueError(f"Invalid logger: {logger}")

        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        if lr is None:
            n_gpus = max(torch.cuda.device_count(), 1)
            lr = 1e-4 * np.sqrt(train_dataloader.batch_size * n_gpus * num_nodes / 128)

        wrapped_model = WrappedModel(self, num_epochs, output, warmup_epochs, lr, nobox, data_gen)

        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator=accelerator,
            precision=precision,
            strategy=strategy,
            num_nodes=num_nodes,
            gradient_clip_val=gradient_clip_val,
            default_root_dir=output if not dry else None,
            logger=logger if not dry else None,
            fast_dev_run=fast_dev_run,
        )
        trainer.fit(
            wrapped_model, train_dataloader, val_dataloader, ckpt_path=ckpt_path
        )
        return

    def extract_levels(self, levels: tuple[float, ...], copy: bool = True) -> "MuViTMAE":
        """Return a new MuViTMAE instance restricted to a subset of levels. Follows the same signature as MuViTEncoder.extract_levels().

        Args:
            levels: Tuple of level scales to keep (must be subset of self.encoder.levels)
            copy: If False and levels match self.encoder.levels, return self without copying

        Returns:
            A new MuViTMAE instance with encoder/decoder restricted to ``levels``.
        """
        if not copy and tuple(levels) == tuple(self.encoder.levels):
            return self

        for level in levels:
            if level not in tuple(self.encoder.levels):
                raise ValueError(f"Requested level {level} not present in encoder levels {self.encoder.levels}")

        level_indices = tuple(self.encoder.levels.index(level) for level in levels)

        new_config = self._config.copy()
        new_config["levels"] = levels
        new_config.pop("ndim", None)

        new_mae = type(self)(**new_config)

        try:
            new_mae.final.load_state_dict(self.final.state_dict())
        except Exception:
            print("Warning: Could not copy final projection layer weights to new MAE.")
            pass

        new_encoder = self.encoder.extract_levels(levels, copy=True)
        new_mae.encoder = new_encoder
       
        if self.decoder_mode == "single":
            new_mae.decoder.load_state_dict(self.decoder.state_dict())
            new_mae.mask_token = nn.Parameter(self.mask_token.data.clone())
        else:
            for new_idx, old_idx in enumerate(level_indices):
                new_mae.decoder[new_idx].load_state_dict(
                    self.decoder[old_idx].state_dict()
                )

            if self.mask_token is not None:
                new_mae.mask_token = nn.Parameter(self.mask_token.data[:, level_indices, :].clone())
        return new_mae
   

class MuViTMAE2d(MuViTMAE[Tuple[int, int]]):
    @classmethod
    @property
    def ndim(self) -> int:
        return 2

    @classmethod
    @property
    def encoder_class(self) -> Type[MuViTEncoder]:
        return MuViTEncoder2d

    @classmethod
    @property
    def decoder_class(self) -> Type[MuViTDecoder]:
        return MuViTDecoder2d

    def patch_token_to_image(self, x: Tensor, C: int, H: int, W: int) -> Tensor:
        """Convert patch tokens back to image.

        Input: (B, N, P*P*C) -> Output: (B, L, C, H, W)
        """
        x = torch.split(x, x.shape[1] // len(self.encoder.levels), dim=1)
        x = torch.stack(
            [
                rearrange(
                    _x,
                    "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                    c=C,
                    h=H // self.patch_size[0],
                    w=W // self.patch_size[1],
                    p1=self.patch_size[0],
                    p2=self.patch_size[1],
                )
                for _x in x
            ],
            dim=1,
        )
        return x


class MuViTMAE3d(MuViTMAE[Tuple[int, int, int]]):
    @classmethod
    @property
    def ndim(self) -> int:
        return 3

    @classmethod
    @property
    def encoder_class(self) -> Type[MuViTEncoder]:
        return MuViTEncoder3d

    @classmethod
    @property
    def decoder_class(self) -> Type[MuViTDecoder]:
        return MuViTDecoder3d

    def patch_token_to_image(self, x: Tensor, C: int, D: int, H: int, W: int) -> Tensor:
        """Convert patch tokens back to image.

        Input: (B, N, D*H*W*C) -> Output: (B, L, C, D, H, W)
        """
        x = torch.split(x, x.shape[1] // len(self.encoder.levels), dim=1)
        x = torch.stack(
            [
                rearrange(
                    _x,
                    "b (d h w) (p1 p2 p3 c) -> b c (d p1) (h p2) (w p3)",
                    c=C,
                    d=D // self.patch_size[0],
                    h=H // self.patch_size[1],
                    w=W // self.patch_size[2],
                    p1=self.patch_size[0],
                    p2=self.patch_size[1],
                    p3=self.patch_size[2],
                )
                for _x in x
            ],
            dim=1,
        )
        return x


class MuViTMAE4d(MuViTMAE[Tuple[int, int, int]]):
    @classmethod
    @property
    def ndim(self) -> int:
        return 4

    @classmethod
    @property
    def encoder_class(self) -> Type[MuViTEncoder]:
        return MuViTEncoder4d

    @classmethod
    @property
    def decoder_class(self) -> Type[MuViTDecoder]:
        return MuViTDecoder4d

    def patch_token_to_image(self, x: Tensor, C: int, T: int, D: int, H: int, W: int) -> Tensor:
        """Convert patch tokens back to image.

        Input: (B, N, T*D*H*W*C) -> Output: (B, L, C, T, D, H, W)
        """
        x = torch.split(x, x.shape[1] // len(self.encoder.levels), dim=1)
        x = torch.stack(
            [
                rearrange(
                    _x,
                    "b (t d h w) (p1 p2 p3 p4 c) -> b c (t p1) (d p2) (h p3) (w p4)",
                    c=C,
                    t=T // self.patch_size[0],
                    d=D // self.patch_size[1],
                    h=H // self.patch_size[2],
                    w=W // self.patch_size[3],
                    p1=self.patch_size[0],
                    p2=self.patch_size[1],
                    p3=self.patch_size[2],
                    p4=self.patch_size[3],
                )
                for _x in x
            ],
            dim=1,
        )
        return x