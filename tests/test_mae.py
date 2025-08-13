import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from typing import Tuple

import pytest
import torch

from muvit.data import MuViTDataset
from muvit.mae import MuViTMAE2d, MuViTMAE3d


class DummyDataset(MuViTDataset):
    def __init__(
        self,
        num_samples: int,
        n_levels: int,
        spatial_size: Tuple[int, int, int],
        n_channels: int,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self._n_levels = n_levels
        self._n_channels = n_channels
        self._ndim = 2 if self.spatial_size[0] == 1 else 3

    @property
    def n_levels(self):
        return self._n_levels

    @property
    def n_channels(self):
        return self._n_channels

    @property
    def ndim(self):
        return self._ndim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate a random image and a random mask
        if idx > len(self):
            raise IndexError("Dataset index out of range")
        img = (
            torch.randn((self.n_levels, self.n_channels, *self.spatial_size))
            .squeeze(2)
            .float()
        )
        bbox = torch.randn((self.n_levels, 2, self.ndim)).float()
        return {
            "img": img,
            "bbox": bbox,
        }


@pytest.mark.parametrize("input_space", ["real", "dct"])
@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("ndim", [2, 3])
def test_mae_fwd(input_space: str, in_channels: int, ndim: int):
    if ndim not in (2, 3):
        raise ValueError("ndim must be either 2 or 3.")
    ActualCls = MuViTMAE3d if ndim == 3 else MuViTMAE2d
    model = ActualCls(
        in_channels=in_channels,
        levels=(1, 4),
        num_layers=3,
        heads=2,
        masking_ratio=0.1,
        dim=320,
        dim_decoder=256,
        input_space=input_space,
    )
    B = 1
    L = 2
    C = in_channels
    H = 64
    W = 64
    if ndim == 2:
        inp = torch.randn((B, L, C, H, W))
    else:
        D = 64
        inp = torch.randn((B, L, C, D, H, W))
    with torch.inference_mode():
        _ = model(inp)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("ndim", [2, 3])
def test_mae_fit(in_channels: int, ndim: int):
    if ndim not in (2, 3):
        raise ValueError("ndim must be either 2 or 3.")
    ActualCls = MuViTMAE2d if ndim == 2 else MuViTMAE3d
    model = ActualCls(
        in_channels=in_channels,
        levels=(1, 4),
        num_layers=3,
        heads=2,
        masking_ratio=0.1,
        dim=320,
        dim_decoder=256,
        input_space="real",
    )
    train_ds = DummyDataset(
        num_samples=16,
        n_levels=2,
        spatial_size=(1, 64, 64) if ndim == 2 else (64, 64, 64),
        n_channels=in_channels,
    )
    val_ds = DummyDataset(
        num_samples=4,
        n_levels=2,
        spatial_size=(1, 64, 64) if ndim == 2 else (64, 64, 64),
        n_channels=in_channels,
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    model.fit(train_dl, val_dl, output=None, dry=True, fast_dev_run=True)


if __name__ == "__main__":
    test_mae_fit(3, 2)
