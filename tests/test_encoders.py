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
        rotary_mode="per_layer",
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
        _ = encoder(inp, return_intermediate_idxs=(0,2))

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
        rotary_mode="per_layer",
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
        rotary_mode="per_layer",
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
        rotary_mode="per_layer",
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


@pytest.mark.parametrize("rotary_mode", ["none", "fixed", "shared", "per_layer"])
def test_rotary_modes_2d(rotary_mode):
    encoder = MuViTEncoder2d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4),
        num_layers=3,
        dim=64,
        heads=2,
        rotary_mode=rotary_mode,
    )
    B, L, C, H, W = 2, 2, 1, 32, 32
    inp = torch.randn((B, L, C, H, W))
    with torch.inference_mode():
        out, coords, level_idx = encoder(inp)
    assert out.shape == (B, L * (H // 4) * (W // 4), 64)

    rope_params = sum(p.numel() for n, p in encoder.named_parameters() if 'rotary_pos_emb' in n)
    if rotary_mode == "none":
        assert rope_params == 0
    elif rotary_mode == "fixed":
        assert rope_params == 0
    elif rotary_mode == "shared":
        assert rope_params == 16
    elif rotary_mode == "per_layer":
        assert rope_params == 48


@pytest.mark.parametrize("rotary_mode", ["none", "fixed", "shared", "per_layer"])
def test_rotary_sharing_2d(rotary_mode):
    encoder = MuViTEncoder2d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4),
        num_layers=3,
        dim=64,
        heads=2,
        rotary_mode=rotary_mode,
    )

    if rotary_mode in ["fixed", "shared"]:
        assert id(encoder.layers[0].rotary_pos_embs) == id(encoder.layers[1].rotary_pos_embs)
        assert id(encoder.layers[1].rotary_pos_embs) == id(encoder.layers[2].rotary_pos_embs)
    elif rotary_mode == "per_layer":
        assert id(encoder.layers[0].rotary_pos_embs) != id(encoder.layers[1].rotary_pos_embs)
        assert id(encoder.layers[1].rotary_pos_embs) != id(encoder.layers[2].rotary_pos_embs)


def test_backward_compat_use_rotary_embed():
    """Test that old saved models with use_rotary_embed can still be loaded."""
    # Test use_rotary_embed=True -> rotary_mode=per_layer
    encoder_true = MuViTEncoder2d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4),
        num_layers=3,
        dim=64,
        heads=2,
        rotary_mode="per_layer",
    )

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model_true"
        encoder_true.save(save_path)

        # Manually modify config to use old use_rotary_embed parameter
        import yaml
        config_path = save_path / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Replace rotary_mode with use_rotary_embed
        config["use_rotary_embed"] = True
        config.pop("rotary_mode")

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Test that loading converts use_rotary_embed to rotary_mode
        loaded_encoder = MuViTEncoder2d.from_folder(save_path)
        assert loaded_encoder._config["rotary_mode"] == "per_layer"
        assert "use_rotary_embed" not in loaded_encoder._config

    # Test use_rotary_embed=False -> rotary_mode=none
    encoder_false = MuViTEncoder2d(
        in_channels=1,
        levels=(1.0, 4.0),
        patch_size=(4, 4),
        num_layers=3,
        dim=64,
        heads=2,
        rotary_mode="none",
    )

    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model_false"
        encoder_false.save(save_path)

        import yaml
        config_path = save_path / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Replace rotary_mode with use_rotary_embed
        config["use_rotary_embed"] = False
        config.pop("rotary_mode")

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded_encoder = MuViTEncoder2d.from_folder(save_path)
        assert loaded_encoder._config["rotary_mode"] == "none"
        assert "use_rotary_embed" not in loaded_encoder._config


if __name__ == "__main__":
    test_encoder_4d()
    print("4d fwd ok")
    test_encoder_io(4)
    print("4d io ok")