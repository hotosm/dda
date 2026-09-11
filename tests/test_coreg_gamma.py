import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dda.pipeline.coreg import _apply_gamma, coregister


def _write_flat(path, value: int, width: int = 32, height: int = 32) -> None:
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 1, 1 / width, 1 / height),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 16,
        "blockysize": 16,
    }
    with rasterio.open(path, "w", **profile) as dst:
        arr = np.full((3, height, width), value, dtype=np.uint8)
        arr[:, 0, 0] = 0
        dst.write(arr)


def test_gamma_lift_brightens_midtones_and_preserves_endpoints(tmp_path):
    p = tmp_path / "flat.tif"
    _write_flat(p, value=128)
    _apply_gamma(p, gamma=0.5, label="pre")
    with rasterio.open(p) as src:
        out = src.read()
    assert out[0, 0, 0] == 0
    # value 128 with gamma 0.5: (128/255)**0.5 * 255 ~= 181
    assert 175 <= int(out[0, 1, 1]) <= 185


def test_gamma_dampen_dims_midtones(tmp_path):
    p = tmp_path / "flat.tif"
    _write_flat(p, value=128)
    _apply_gamma(p, gamma=1.5, label="pre")
    with rasterio.open(p) as src:
        out = src.read()
    # value 128 with gamma 1.5: (128/255)**1.5 * 255 ~= 90
    assert 85 <= int(out[0, 1, 1]) <= 95


def test_gamma_preserves_highlights(tmp_path):
    p = tmp_path / "flat.tif"
    _write_flat(p, value=255)
    _apply_gamma(p, gamma=0.85, label="post")
    with rasterio.open(p) as src:
        out = src.read()
    # 255 stays 255 for any gamma
    assert int(out[0, 5, 5]) == 255


def test_gamma_handles_multi_block_and_edge_trim(tmp_path, monkeypatch):
    from dda.pipeline import coreg as coreg_mod

    monkeypatch.setattr(coreg_mod, "GAMMA_BLOCK_PX", 16)
    p = tmp_path / "big.tif"
    _write_flat(p, value=128, width=40, height=25)
    _apply_gamma(p, gamma=0.5, label="pre")
    with rasterio.open(p) as src:
        out = src.read()
    assert out.shape == (3, 25, 40)
    assert out[0, 0, 0] == 0
    lifted = out[:, 0, 1:]
    assert lifted.min() >= 175 and lifted.max() <= 185
    # last row + column proves edge blocks were written
    assert 175 <= int(out[0, 24, 39]) <= 185


def test_coregister_rejects_out_of_range_gamma(tmp_path):
    with pytest.raises(ValueError, match="pre_gamma"):
        coregister(
            pre_raw=tmp_path / "pre.tif",
            post_raw=tmp_path / "post.tif",
            pre_aligned=tmp_path / "pre_aligned.tif",
            post_aligned=tmp_path / "post_aligned.tif",
            drift_json=tmp_path / "drift.json",
            check_png=tmp_path / "check.png",
            pre_gamma=0.0,
        )
    with pytest.raises(ValueError, match="post_gamma"):
        coregister(
            pre_raw=tmp_path / "pre.tif",
            post_raw=tmp_path / "post.tif",
            pre_aligned=tmp_path / "pre_aligned.tif",
            post_aligned=tmp_path / "post_aligned.tif",
            drift_json=tmp_path / "drift.json",
            check_png=tmp_path / "check.png",
            post_gamma=10.0,
        )
