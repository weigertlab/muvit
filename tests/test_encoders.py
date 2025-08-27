import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pytest
import torch
from muvit.encoders import MuViTEncoder2d, MuViTEncoder3d, MuViTEncoder4d
from tempfile import TemporaryDirectory
from pathlib import Path


NDIM_TO_ENCODER_CLS = {
    2: MuViTEncoder2d,
    3: MuViTEncoder3d,
    4: MuViTEncoder4d
}

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


def test_encoder_4d():
    encoder = MuViTEncoder4d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4, 4, 4),
        num_layers=3,
        dim=128,
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
    T = 8
    D = 16
    H = 16
    W = 16

    inp = torch.randn((B, L, C, T, D, H, W))
    print(inp.shape)
    with torch.inference_mode():
        _ = encoder(inp)

    C = 2
    inp = torch.randn((B, L, C, T, D, H, W))
    with pytest.raises(RuntimeError):
        with torch.inference_mode():
            _ = encoder(inp)



@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_encoder_io(ndim: int):
    ActualCls = NDIM_TO_ENCODER_CLS[ndim]
    OtherCls = NDIM_TO_ENCODER_CLS[(ndim+1) if ndim != 4 else 2]

    encoder = ActualCls(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4,)*ndim,
        num_layers=3,
        dim=128,
        heads=2,
        attention_mode="all",
        use_level_embed=True,
        use_rotary_embed=True,
        input_space="real",
        dropout=0.0,
    )
    with TemporaryDirectory() as tmpdir:
        encoder.save(Path(tmpdir)/"model", overwrite=True)

        # wrong ndim model
        with pytest.raises(ValueError):
            _ = OtherCls.from_folder(Path(tmpdir)/"model")

        loaded_encoder = ActualCls.from_folder(Path(tmpdir)/"model")
    assert encoder._config == loaded_encoder._config
    assert all((p1==p2).all() for p1, p2 in zip(encoder.parameters(), loaded_encoder.parameters()))


if __name__ == "__main__":
    test_encoder_4d()
    print("4d fwd ok")
    test_encoder_io(4)
    print("4d io ok")