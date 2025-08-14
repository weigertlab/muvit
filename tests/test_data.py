import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pytest
import torch
from muvit.data import MuViTDataset
from pathlib import Path

from typing import Tuple

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
        self._levels = list(range(1, n_levels+1))
        self._n_channels = n_channels
        self._ndim = 2 if self.spatial_size[0] == 1 else 3

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
            .squeeze(2)
            .float()
        )
        bbox = torch.randn((self.n_levels, 2, self.ndim)).float()
        return {
            "img": img,
            "bbox": bbox,
        }
    
class BuggyDummyDataset(DummyDataset):
    def __init__(self, bug_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bug_type = bug_type
    
    def __getitem__(self, idx: int):
        res = super().__getitem__(idx)
        if self._bug_type == "img_shape":
            res["img"] = res["img"].unsqueeze(0)
        elif self._bug_type == "img_lvs":
            res["img"] = torch.stack(2*[res["img"]], dim=0)
        elif self._bug_type == "img_ch":
            res["img"] = torch.stack(2*[res["img"]], dim=0)
        elif self._bug_type == "bbox_shape":
            res["bbox"] = res["bbox"].unsqueeze(0)
        elif self._bug_type == "bbox_lvs":
            res["bbox"] = torch.stack(2*[res["bbox"]], dim=0)
        elif self._bug_type == "bbox_ndim":
            res["bbox"] = torch.stack(2*[res["bbox"]], dim=2)
        return res

@pytest.mark.parametrize("ndim", [2, 3])
def test_data_ok(ndim: int):
    ds = DummyDataset(
        num_samples=10,
        n_levels=2,
        spatial_size=(1, 32, 32) if ndim == 2 else (16, 32, 32),
        n_channels=1,
    )
    _item = ds[3]
    assert "img" in _item.keys()
    assert "bbox" in _item.keys()

@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("bug_type", ["img_shape", "img_lvs", "img_ch", "bbox_shape", "bbox_lvs", "bbox_ndim"])
def test_data_buggy(ndim: int, bug_type: str):
    with pytest.raises(ValueError):
        _ = BuggyDummyDataset(
            bug_type=bug_type,
            num_samples=10,
            n_levels=2,
            spatial_size=(1, 32, 32) if ndim == 2 else (16, 32, 32),
            n_channels=1,
        )

if __name__ == "__main__":
    ds = DummyDataset(10, 2, (1,32,32),1)