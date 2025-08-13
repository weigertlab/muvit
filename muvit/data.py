from torch.utils.data import Dataset
from abc import ABC, ABCMeta, abstractmethod


class SanityCheckMeta(ABCMeta):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        if hasattr(obj, "_sanity_check"):
            obj._sanity_check()
        return obj


class MuViTDataset(Dataset, ABC, metaclass=SanityCheckMeta):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __getitem__(self, idx: int):
        pass
    
    @abstractmethod
    def __len__(self):
        pass
    
    @property
    @abstractmethod
    def ndim(self):
        pass

    @property
    @abstractmethod
    def n_levels(self):
        pass

    @property
    @abstractmethod
    def n_channels(self):
        pass

    def _sanity_check(self):
        if self.ndim not in (2, 3):
            raise ValueError("ndim must be 2 or 3")
        if len(self) == 0:
            raise ValueError("Dataset is empty.")
        sample = self[0]
        if "img" not in sample or "bbox" not in sample:
            raise ValueError("Sample must contain 'img' and 'bbox' keys.")

        img = sample["img"]
        bbox = sample["bbox"]

        if bbox is not None:
            expected_bbox_shape = (self.n_levels, 2, self.ndim)
            if tuple(bbox.shape) != expected_bbox_shape:
                raise ValueError(f"Expected bbox shape {expected_bbox_shape}, got {tuple(bbox.shape)}")

        expected_img_ndim = 4 if self.ndim == 2 else 5
        if img.ndim != expected_img_ndim:
            raise ValueError(
                f"Expected img.ndim = {expected_img_ndim} for ndim={self.ndim} (L,C"
                f"{',D' if self.ndim==3 else ''},H,W), got {img.ndim}"
            )

        if img.shape[0] != self.n_levels:
            raise ValueError(f"Expected img.shape[0] (L) = {self.n_levels}, got {img.shape[0]}")
        if img.shape[1] != self.n_channels:
            raise ValueError(f"Expected img.shape[1] (C) = {self.n_channels}, got {img.shape[1]}")