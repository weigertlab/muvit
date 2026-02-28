import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import pytest
import torch
from muvit.data import MuViTDataset
from muvit.mae import MuViTMAE2d, MuViTMAE3d, MuViTMAE4d

NDIM_TO_MAE_CLS = {
    2: MuViTMAE2d,
    3: MuViTMAE3d,
    4: MuViTMAE4d
}

class DummyDataset(MuViTDataset):
    def __init__(
        self,
        num_samples: int,
        n_levels: int,
        spatial_size: Tuple[int, int, int, int],
        n_channels: int,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.spatial_size = spatial_size
        self._levels = list(range(1, n_levels+1))
        self._n_channels = n_channels
        self._ndim = len(spatial_size)

    @property
    def levels(self):
        return self._levels

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
            .float()
        )
        bbox = torch.randn((self.n_levels, 2, self.ndim)).float()
        return {
            "img": img,
            "bbox": bbox,
        }


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_mae_fwd(input_space: str, in_channels: int, ndim: int):
    if ndim not in (2, 3, 4):
        raise ValueError("ndim must be either 2, 3, or 4.")
    ActualCls = NDIM_TO_MAE_CLS[ndim]

    model = ActualCls(
        in_channels=in_channels,
        levels=(1, 4),
        num_layers=3,
        heads=2,
        masking_ratio=0.1,
        dim=320,
        dim_decoder=256,
        patch_size=4,
    )
    B = 1
    L = 2
    C = in_channels
    H = 32
    W = 32
    if ndim == 2:
        inp = torch.randn((B, L, C, H, W))
    elif ndim == 3:
        D = 16
        inp = torch.randn((B, L, C, D, H, W))
    elif ndim == 4:
        D = 16
        T = 4
        inp = torch.randn((B, L, C, T, D, H, W))
    with torch.inference_mode():
        _ = model(inp)


@pytest.mark.parametrize("in_channels", [1, 3])
@pytest.mark.parametrize("ndim", [2, 3, 4])
@pytest.mark.filterwarnings("ignore:Starting from*") # lightning logging warning
@pytest.mark.filterwarnings("ignore:The '*") # lightning dataloader warning
def test_mae_fit(in_channels: int, ndim: int):
    if ndim not in (2, 3, 4):
        raise ValueError("ndim must be either 2 or 3.")
    ActualCls = ActualCls = NDIM_TO_MAE_CLS[ndim]

    model = ActualCls(
        in_channels=in_channels,
        levels=(1, 4),
        num_layers=3,
        heads=2,
        masking_ratio=0.1,
        dim=320,
        dim_decoder=256,
        input_space="real",
        patch_size=4,
    )
    dim_to_spatial_size = {
        2: (32,32),
        3: (16,32,32),
        4: (4,16,32,32)
    }
    train_ds = DummyDataset(
        num_samples=16,
        n_levels=2,
        spatial_size=dim_to_spatial_size[ndim],
        n_channels=in_channels,
    )
    val_ds = DummyDataset(
        num_samples=4,
        n_levels=2,
        spatial_size=dim_to_spatial_size[ndim],
        n_channels=in_channels,
    )
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)
    model.fit(train_dl, val_dl, output=None, dry=True, fast_dev_run=True)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_mae_io(ndim: int):
    ActualCls = NDIM_TO_MAE_CLS[ndim]
    OtherCls = NDIM_TO_MAE_CLS[(ndim+1) if ndim != 4 else 2]
    model = ActualCls(
        in_channels=1,
        levels=(1, 4),
        num_layers=3,
        heads=2,
        masking_ratio=0.1,
        dim=128,
        dim_decoder=128,
        input_space="real",
        patch_size=4,
    )
    with TemporaryDirectory() as tmpdir:
        model.save(Path(tmpdir) / "model", overwrite=True)

        # Load wrong ndim model
        with pytest.raises(ValueError):
            _ = OtherCls.from_folder(Path(tmpdir) / "model")

        loaded_model = ActualCls.from_folder(Path(tmpdir) / "model")
    assert model._config == loaded_model._config
    assert all(
        (p1 == p2).all()
        for p1, p2 in zip(model.parameters(), loaded_model.parameters())
    )

if __name__ == "__main__":
    test_mae_fwd("real", 1, 4)