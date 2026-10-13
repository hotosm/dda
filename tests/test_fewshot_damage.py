"""Tests for `dda.pipeline.fewshot_damage`: label parsing, chip windowing, freeze scope.

Fast tests only. The full Lightning training loop needs the DINOv3-L backbone + damage
checkpoint from HF (2+ GB) and a GPU to be realistic, so a real-fit smoke lives under the
`slow` marker in `test_train_damage_fewshot_slow` (opt-in via `pytest -m slow`).
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from dda.pipeline import fewshot_damage as fd


def _write_geotiff(path: Path, arr: np.ndarray, transform, crs="EPSG:32619") -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr)


def _write_labels(path: Path, polys_with_class: list[tuple[Polygon, int | str]], crs="EPSG:32619") -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"damage": [c for _, c in polys_with_class]},
        geometry=[p for p, _ in polys_with_class],
        crs=crs,
    )
    gdf.to_file(path, driver="GeoJSON")


class TestNormalizeDamageValue:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1, 1),
            (2, 2),
            (3, 3),
            (4, 4),
            ("no-damage", 1),
            ("no_damage", 1),
            ("minor-damage", 2),
            ("MINOR", 2),
            ("major-damage", 3),
            ("major", 3),
            ("destroyed", 4),
            ("Destroyed", 4),
        ],
    )
    def test_valid_values(self, value, expected):
        assert fd.normalize_damage_value(value) == expected

    @pytest.mark.parametrize("bad", ["totalled", 0, ""])
    def test_rejects_unknown(self, bad):
        with pytest.raises((ValueError, KeyError)):
            fd.normalize_damage_value(bad)


class TestReadLabels:
    def test_missing_damage_column_raises(self, tmp_path):
        import geopandas as gpd

        p = tmp_path / "labels.geojson"
        gpd.GeoDataFrame(
            {"foo": [1]}, geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])], crs="EPSG:4326"
        ).to_file(p, driver="GeoJSON")
        with pytest.raises(ValueError, match="`damage` column"):
            fd._read_labels(p)

    def test_reads_mixed_int_and_string(self, tmp_path):
        p = tmp_path / "labels.geojson"
        _write_labels(
            p,
            [
                (Polygon([(0, 0), (10, 0), (10, 10), (0, 0)]), 4),
                (Polygon([(20, 0), (30, 0), (30, 10), (20, 0)]), "minor-damage"),
            ],
            crs="EPSG:4326",
        )
        result = fd._read_labels(p)
        assert [c for _, c in result] == [4, 2]


class TestChipArea:
    def _make_scene(self, tmp_path: Path, aoi_size_m: int = 2048):
        """Small synthetic scene: 1 m/pixel, EPSG:32619, all-ones RGB."""
        tf = from_origin(500000, 4500000, 1.0, 1.0)
        arr = np.full((3, aoi_size_m, aoi_size_m), 128, dtype=np.uint8)
        pre = tmp_path / "pre.tif"
        post = tmp_path / "post.tif"
        _write_geotiff(pre, arr, tf)
        _write_geotiff(post, arr, tf)
        return pre, post, tf

    def test_returns_only_chips_with_buildings(self, tmp_path):
        pre, post, _tf = self._make_scene(tmp_path, aoi_size_m=1024)
        b1 = Polygon([(500100, 4499900), (500110, 4499900), (500110, 4499890), (500100, 4499890)])
        labels = tmp_path / "labels.geojson"
        _write_labels(labels, [(b1, 3)])
        chips = fd._chip_area(pre, post, labels, tile_size=256, stride=256)
        assert len(chips) >= 1
        for c in chips:
            assert int((c["damage"] > 0).sum()) > 0

    def test_no_buildings_returns_empty(self, tmp_path):
        pre, post, _tf = self._make_scene(tmp_path, aoi_size_m=512)
        far = Polygon([(600000, 4400000), (600010, 4400000), (600010, 4400010), (600000, 4400010)])
        labels = tmp_path / "labels.geojson"
        _write_labels(labels, [(far, 1)])
        chips = fd._chip_area(pre, post, labels, tile_size=256, stride=256)
        assert chips == []

    def test_skips_chips_with_thin_pre_coverage(self, tmp_path):
        tf = from_origin(500000, 4500000, 1.0, 1.0)
        pre_arr = np.zeros((3, 512, 512), dtype=np.uint8)
        post_arr = np.full((3, 512, 512), 128, dtype=np.uint8)
        pre = tmp_path / "pre.tif"
        post = tmp_path / "post.tif"
        _write_geotiff(pre, pre_arr, tf)
        _write_geotiff(post, post_arr, tf)
        b = Polygon([(500100, 4499900), (500110, 4499900), (500110, 4499890), (500100, 4499890)])
        labels = tmp_path / "labels.geojson"
        _write_labels(labels, [(b, 4)])
        chips = fd._chip_area(pre, post, labels, tile_size=256, stride=256)
        assert chips == []


class TestStrictHeadFreeze:
    def test_freezes_decoder_pyramid_fusion_only(self):
        from torch import nn

        class _FakeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(4, 4)
                self.pyramid = nn.Linear(4, 4)
                self.decoder = nn.Linear(4, 4)
                self.fusion = nn.Linear(4, 4)
                self.loc_head = nn.Linear(4, 1)
                self.dmg_head = nn.Linear(4, 4)

        class _FakeModel:
            def __init__(self):
                self.net = _FakeNet()

        m = _FakeModel()
        for p in m.net.backbone.parameters():
            p.requires_grad = False

        n_train, n_frozen = fd._apply_strict_head_freeze(m)  # ty: ignore[invalid-argument-type]

        for name in ("backbone", "pyramid", "decoder", "fusion"):
            module = getattr(m.net, name)
            assert all(not p.requires_grad for p in module.parameters()), f"{name} not fully frozen"
        for name in ("loc_head", "dmg_head"):
            module = getattr(m.net, name)
            assert all(p.requires_grad for p in module.parameters()), f"{name} unexpectedly frozen"

        expected_trainable = sum(p.numel() for p in m.net.loc_head.parameters()) + sum(
            p.numel() for p in m.net.dmg_head.parameters()
        )
        assert n_train == expected_trainable
        assert n_frozen > n_train

    def test_handles_fusion_is_none(self):
        from torch import nn

        class _FakeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(4, 4)
                self.pyramid = nn.Linear(4, 4)
                self.decoder = nn.Linear(4, 4)
                self.fusion = None
                self.loc_head = nn.Linear(4, 1)
                self.dmg_head = nn.Linear(4, 4)

        class _FakeModel:
            def __init__(self):
                self.net = _FakeNet()

        m = _FakeModel()
        for p in m.net.backbone.parameters():
            p.requires_grad = False
        n_train, _ = fd._apply_strict_head_freeze(m)  # ty: ignore[invalid-argument-type]
        assert n_train > 0


@pytest.mark.slow
def test_train_damage_fewshot_slow(tmp_path):
    """Real end-to-end fit on a synthetic scene. Downloads the damage checkpoint (~500 MB)."""
    tf = from_origin(500000, 4500000, 1.0, 1.0)
    arr = np.full((3, 2048, 2048), 128, dtype=np.uint8)
    pre = tmp_path / "pre.tif"
    post = tmp_path / "post.tif"
    _write_geotiff(pre, arr, tf)
    _write_geotiff(post, arr + 20, tf)
    labels = tmp_path / "labels.geojson"
    _write_labels(
        labels,
        [
            (
                Polygon(
                    [
                        (500000 + x, 4500000 - y),
                        (500000 + x + 12, 4500000 - y),
                        (500000 + x + 12, 4500000 - y - 12),
                        (500000 + x, 4500000 - y - 12),
                    ]
                ),
                3,
            )
            for x in range(60, 1900, 200)
            for y in range(60, 1900, 200)
        ],
    )
    result = fd.fit_damage(
        pre,
        post,
        labels,
        tmp_path / "out",
        epochs=1,
        batch_size=2,
        tile_size=256,
        stride=256,
    )
    assert result.n_train >= 2
    assert result.best_ckpt.exists()


def test_fit_damage_windows_rejects_a_single_validation_window(tmp_path):
    window = {
        "post": np.zeros((3, 32, 32), dtype=np.uint8),
        "pre": np.zeros((3, 32, 32), dtype=np.uint8),
        "damage": np.ones((32, 32), dtype=np.uint8),
    }
    with pytest.raises(RuntimeError, match="at least 2 train"):
        fd.fit_damage_windows([window, window], [window], tmp_path)
