import pytest
import torch
from muvit.decoders import MuViTDecoder2d, MuViTDecoder3d, MuViTDecoder4d


def test_decoder_2d():
    decoder = MuViTDecoder2d(
        in_channels=64,
        dim=64,
        num_layers=3,
        heads=2,
        use_rotary_embed=True,
        dropout=0.0,
    )
    B = 2
    N = 128
    D = 64

    inp = torch.randn((B, N, D))
    with torch.inference_mode():
        _ = decoder(inp)


def test_decoder_3d():
    decoder = MuViTDecoder3d(
        in_channels=64,
        dim=64,
        num_layers=3,
        heads=2,
        use_rotary_embed=True,
        dropout=0.0,
    )
    B = 2
    N = 128
    D = 64

    inp = torch.randn((B, N, D))
    with torch.inference_mode():
        _ = decoder(inp)

def test_decoder_4d():
    decoder = MuViTDecoder4d(
        in_channels=64,
        dim=128,
        num_layers=3,
        heads=2,
        use_rotary_embed=True,
        dropout=0.0,
    )
    B = 2
    N = 128
    D = 64

    inp = torch.randn((B, N, D))
    with torch.inference_mode():
        _ = decoder(inp)
