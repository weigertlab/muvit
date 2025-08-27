import io

import numpy as np
import torch
from PIL import Image
from skimage import draw


def compress_pil(im: np.ndarray, width: int = 512, quality=75):
    """im.shape = (H, W, C)"""
    if im.dtype != np.uint8:
        im = (np.clip(im, 0, 1) * 255).astype(np.uint8)

    pil_image = Image.fromarray(im)
    w, h = pil_image.size
    new_height = int(h * width / w)
    pil_image = pil_image.resize((width, new_height))
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)


def compute_relative_inset(box1, box2, shape=(1.0, 1.0)):
    """
    Compute the relative inset of box1 in box2 (wrt shape)
    """
    assert box1.shape == (2, 2) and box2.shape == (2, 2)

    if isinstance(box1, torch.Tensor):
        box1 = box1.detach().cpu().numpy()
    if isinstance(box2, torch.Tensor):
        box2 = box2.detach().cpu().numpy()

    b2_size = box2[1] - box2[0]
    scale = np.array(shape) / b2_size
    b1_rel = box1 - box2[0]
    b1_scaled = b1_rel * scale
    return b1_scaled


def box_annotate(x, bbox, lines=True):
    L, C, H, W = x.shape
    assert bbox.shape == (L, 2, 2)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().numpy()

    for i, (box1, box2) in enumerate(zip(bbox[:-1], bbox[1:])):
        box_rel = compute_relative_inset(box1, box2, shape=(H, W))
        rr, cc = draw.rectangle_perimeter(box_rel[0], box_rel[1], shape=(H, W))
        x[i + 1, :, rr, cc] = 1
        if lines:
            rr, cc, val = draw.line_aa(0, 0, int(box_rel[0][0]), int(box_rel[0][1]))
            x[i + 1, :, rr, cc] = np.repeat(val[:, None], C, axis=-1)
            rr, cc, val = draw.line_aa(0, W - 1, int(box_rel[0][0]), int(box_rel[1][1]))
            x[i + 1, :, rr, cc] = np.repeat(val[:, None], C, axis=-1)
    return x


def create_image_grid(x, output, bbox=None):
    # L, C, H, W
    if x.ndim == 5:
        B, L, C, H, W = x.shape
        D = None
        T = None
    elif x.ndim == 6:
        B, L, C, D, H, W = x.shape
        T = None
    elif x.ndim == 7:
        B, L, C, T, D, H, W = x.shape
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    x = x[0].detach().cpu().numpy()
    y = output["reco"][0].detach().cpu().numpy()
    z = output["input_masked"][0].detach().cpu().numpy()

    if T is not None and D is not None:
        x = x[:, :, T // 2, D // 2]
        y = y[:, :, T // 2, D // 2]
        z = z[:, :, T // 2, D // 2]
        bbox = bbox[..., 2:]

    elif T is None and D is not None:
        x = x[:, :, D // 2]
        y = y[:, :, D // 2]
        z = z[:, :, D // 2]
        bbox = bbox[..., 1:]

    err = np.abs(x - y)

    if bbox is not None:
        box_annotate(x, bbox, lines=False)

    x = np.concatenate(
        np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=1), axis=-2
    )
    y = np.concatenate(
        np.pad(y, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=1), axis=-2
    )
    z = np.concatenate(
        np.pad(z, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=1), axis=-2
    )
    err = np.concatenate(
        np.pad(err, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=1), axis=-2
    )
    out = np.concatenate([z, y, x, err], axis=-1)

    if out.shape[0] == 1:
        out = out[0]
    elif out.shape[0] == 2:
        # rgb
        out = np.stack([out[0], out[1], out[0]], axis=0)
    elif out.shape[0] > 3:
        out = out[:3]
    out = np.clip(out, 0, 1)
    return out
