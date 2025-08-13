import pytest
import torch
from muvit.mae import MuViTMAE2d, MuViTMAE3d


@pytest.mark.parametrize("input_space", ["real", "dct"])
@pytest.mark.parametrize("in_channels", [1, 3])
def test_mae_2d(input_space: str, in_channels: int):
    model = MuViTMAE2d(
        in_channels=in_channels,
        levels=(1,4),
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

    inp = torch.randn((B, L, C, H, W))

    with torch.inference_mode():
        _ = model(inp)

@pytest.mark.parametrize("input_space", ["real", "dct"])
@pytest.mark.parametrize("in_channels", [1, 3])
def test_mae_3d(input_space: str, in_channels: int):
    model = MuViTMAE3d(
        in_channels=in_channels,
        levels=(1,4),
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
    D = 64
    H = 64
    W = 64

    inp = torch.randn((B, L, C, D, H, W))
    with torch.inference_mode():
        _ = model(inp)
