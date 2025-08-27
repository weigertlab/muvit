from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset


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
    def n_levels(self):
        return len(self.levels)

    @property
    @abstractmethod
    def levels(self):
        pass

    @property
    @abstractmethod
    def n_channels(self):
        pass

    def _sanity_check(self):
        if self.ndim not in (2, 3, 4):
            raise ValueError("ndim must be 2, 3 or 4")
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
                raise ValueError(
                    f"Expected bbox shape {expected_bbox_shape}, got {tuple(bbox.shape)}"
                )

        expected_img_ndim = {
            2: 4,
            3: 5,
            4: 6
        }[self.ndim]
        if img.ndim != expected_img_ndim:
            raise ValueError(
                f"Expected img.ndim = {expected_img_ndim} for ndim={self.ndim} (L,C"
                f"{',D' if self.ndim == 3 else ''},H,W), got {img.ndim}"
            )

        if img.shape[0] != self.n_levels:
            raise ValueError(
                f"Expected img.shape[0] (L) = {self.n_levels}, got {img.shape[0]}"
            )
        if img.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected img.shape[1] (C) = {self.n_channels}, got {img.shape[1]}"
            )

    def visualize_sample(self, idx: int = 0, save_file: Optional[Path] = None):
        if self.ndim != 2:
            raise NotImplementedError(
                "Sample visualization in 3d is not implemented yet."
            )
        import numpy as np
        from .utils import box_annotate

        _item = self[idx]
        img, bbox = _item["img"].cpu().numpy(), _item["bbox"].cpu().numpy()

        box_annotate(img, bbox)
        img = np.concatenate(
            np.pad(img, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=1), axis=-2
        )

        if img.shape[0] == 1:
            img = img[0]
        elif img.shape[0] == 2:
            img = np.stack([img[0], img[1], img[0]], axis=0)
        elif img.shape[0] > 3:
            img = img[:3]

        if save_file:
            import matplotlib.pyplot as plt
            save_file.parent.mkdir(exist_ok=True, parents=True)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(img.transpose(1, 2, 0))
            ax.axis("off")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            fig.savefig(save_file, bbox_inches="tight")
            plt.close(fig)
