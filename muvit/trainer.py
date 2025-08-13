from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import lightning.pytorch as pl
import numpy as np
import torch
import wandb
from lightning.pytorch.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from .utils import compress_pil, create_image_grid

if TYPE_CHECKING:
    from .mae import MuViTMAE


class WrappedModel(pl.LightningModule):
    def __init__(
        self,
        model: MuViTMAE,
        num_epochs: int,
        outdir: Union[Path, None],
        warmup_epochs: int = 5,
        learning_rate: float = 1e-4,
        nobox: bool = False,
    ):
        super().__init__()
        self.model = model
        self.max_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.learning_rate = learning_rate
        self.outdir = outdir
        self.nobox = nobox
        self.save_hyperparameters()
        self.data_gen = torch.Generator()

    def on_train_batch_start(self, batch, batch_idx):
        # Access the learning rate from the optimizer
        optimizer = self.trainer.optimizers[0]  # Assuming one optimizer
        lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", lr, on_step=True, on_epoch=False, prog_bar=False)

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero and self.outdir is not None:
            self.model.save(self.outdir, overwrite=True)

    def on_train_epoch_start(self):
        epoch_seed = 42 + self.current_epoch
        self.data_gen.manual_seed(epoch_seed)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, torch.Tensor) and batch.ndim == 5:
            bbox = None
        elif isinstance(batch, dict):
            bbox = batch.get("bbox", None)
            img = batch["img"]
        elif (
            len(batch) == 2
            and isinstance(batch[0], torch.Tensor)
            and batch[0].ndim == 5
        ):
            bbox = batch[1]
            img = batch[0]
        else:
            raise ValueError(
                f"Invalid batch format. Expected either a dictionary, a tuple (X,bbox) or a 5D tensor X, got {type(batch)}"
            )

        output = self.model(img, bbox=bbox if not self.nobox else None)
        loss = output["loss"]
        if torch.isnan(loss):
            print(f"Loss is NaN at step {self.global_step}")
            print(f"Batch shape: {img.shape}")
            print(f"Output: {output}")
            raise ValueError("Loss is NaN")

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("batch_size", len(img), on_step=True, prog_bar=False, sync_dist=False)
        self.log(
            "batch_mean",
            img.mean().item(),
            on_step=True,
            prog_bar=False,
            sync_dist=False,
        )
        _norm = grad_norm(self.model, 2).get("grad_2.0_norm_total", 0)
        self.log("grad_norm", _norm, on_step=True, prog_bar=False, sync_dist=False)

        if (
            self.current_epoch % 2 == 0
            and self.trainer.is_global_zero
            and batch_idx == 0
            and not self.trainer.sanity_checking
        ):
            with torch.inference_mode():
                img = img[:1]
                if bbox is not None:
                    bbox = bbox[:1]
                output = self.model(img, bbox=bbox, return_all=True)
                self.log_images(img, output, bbox, batch_idx, name="train_inout")

        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, torch.Tensor) and batch.ndim == 5:
            bbox = None
        elif isinstance(batch, dict):
            bbox = batch.get("bbox", None)
            img = batch["img"]
        elif (
            len(batch) == 2
            and isinstance(batch[0], torch.Tensor)
            and batch[0].ndim == 5
        ):
            bbox = batch[1]
            img = batch[0]
        else:
            raise ValueError(
                f"Invalid batch format. Expected either a dictionary, a tuple (X,bbox) or a 5D tensor X, got {type(batch)}"
            )
        output = self.model(img, bbox=bbox if not self.nobox else None, return_all=True)
        loss = output["loss"]
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if (
            self.trainer.is_global_zero
            and batch_idx < 4
            and not self.trainer.sanity_checking
        ):
            self.log_images(img, output, bbox, batch_idx, name="val_inout")
            if self.current_epoch % 10 == 0:
                self.log_images(
                    img,
                    output,
                    bbox,
                    batch_idx,
                    name="val_bigger",
                    width=1024,
                    quality=95,
                )

        return loss

    def log_images(self, img, output, bbox, batch_idx, name, width=512, quality=95):
        out = create_image_grid(img, output, bbox[0] if bbox is not None else None)

        if isinstance(self.logger, pl.loggers.TensorBoardLogger):
            self.logger.experiment.add_images(
                f"{name}{batch_idx}", out[None], global_step=self.global_step
            )

        elif isinstance(self.logger, pl.loggers.WandbLogger):
            if out.ndim == 3:
                out = np.moveaxis(out, 0, -1)

            _img = wandb.Image(compress_pil(out, width=width, quality=quality))
            self.logger.experiment.log({f"{name}{batch_idx}": _img})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), betas=(0.9, 0.95), lr=self.learning_rate
        )
        scheduler = [
            {
                "scheduler": CosineAnnealingLR(optimizer, T_max=self.max_epochs),
            }
        ]
        scheduler.append(
            {
                "scheduler": LinearLR(
                    optimizer,
                    start_factor=1e-2,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                ),
            }
        )
        return [optimizer], scheduler
