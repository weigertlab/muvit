import pytest

from muvit.encoders import MuViTEncoder2d, MuViTEncoder3d
import torch


def test_encoder_2d():
    encoder = MuViTEncoder2d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4),
        num_layers=3,
        dim=64,
        heads=2,
        attention_mode="all",
        use_level_embed=True,
        use_rotary_embed=True,
        input_space="real",
        dropout=0.0,
    )
    B = 2
    L = 2
    C = 1
    H = 32
    W = 32

    inp = torch.randn((B, L, C, H, W))
    with torch.inference_mode():
        _ = encoder(inp)

    C = 2
    inp = torch.randn((B, L, C, H, W))
    with pytest.raises(RuntimeError):
        with torch.inference_mode():
            _ = encoder(inp)


def test_encoder_3d():
    encoder = MuViTEncoder3d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4, 4),
        num_layers=3,
        dim=64,
        heads=2,
        attention_mode="all",
        use_level_embed=True,
        use_rotary_embed=True,
        input_space="real",
        dropout=0.0,
    )
    B = 2
    L = 2
    C = 1
    D = 16
    H = 32
    W = 32

    inp = torch.randn((B, L, C, D, H, W))
    with torch.inference_mode():
        _ = encoder(inp)

    C = 2
    inp = torch.randn((B, L, C, D, H, W))
    with pytest.raises(RuntimeError):
        with torch.inference_mode():
            _ = encoder(inp)
