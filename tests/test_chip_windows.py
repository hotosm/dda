from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import pytest
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import from_bounds
from shapely.geometry import box

from dda.pipeline.chip_windows import area_inputs, parse_chip_name, windows_from_chips

ZOOM = 18
CHIP = 256
X, Y = 193232, 109610


def _write_chip(directory: Path, x: int, y: int, seed: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    west, south, east, north = mercantile.bounds(x, y, ZOOM)
    name = f"OAM-{x}-{y}-{ZOOM}.tif"
    rng = np.random.default_rng(seed)
    with rasterio.open(
        directory / name,
        "w",
        driver="GTiff",
        width=CHIP,
        height=CHIP,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(west, south, east, north, CHIP, CHIP),
    ) as dst:
        dst.write(rng.integers(40, 220, size=(3, CHIP, CHIP), dtype=np.uint8))
    return name


def _tile_center_square(x: int, y: int, fraction: float = 0.2):
    west, south, east, north = mercantile.bounds(x, y, ZOOM)
    cx, cy = (west + east) / 2, (south + north) / 2
    half_w, half_h = (east - west) * fraction / 2, (north - south) * fraction / 2
    return box(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _labels(rows: Sequence[tuple[object, object]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"damage": [damage for _, damage in rows]}, geometry=[g for g, _ in rows], crs="EPSG:4326"
    )


def test_parse_chip_name_reads_tile_coordinates():
    assert parse_chip_name("OAM-101-200-18.tif") == (101, 200, 18)
    with pytest.raises(ValueError, match="does not match"):
        parse_chip_name("tile_101_200.tif")


def test_windows_keep_only_labelled_blocks_with_labels_on_the_right_tile(tmp_path):
    tiles = [(X, Y), (X + 1, Y), (X + 4, Y)]
    names = [_write_chip(tmp_path / "post", x, y, seed=i) for i, (x, y) in enumerate(tiles)]
    for i, (x, y) in enumerate(tiles):
        _write_chip(tmp_path / "pre", x, y, seed=10 + i)
    labels = _labels([(_tile_center_square(X + 1, Y), "destroyed")])

    windows = windows_from_chips(tmp_path / "post", tmp_path / "pre", labels, names)

    assert len(windows) == 1
    window = windows[0]
    assert window["post"].shape == window["pre"].shape == (3, 2 * CHIP, 2 * CHIP)
    damage = window["damage"]
    assert set(np.unique(damage)) == {0, 4}
    assert damage[:CHIP, CHIP:].any()
    assert not damage[CHIP:, :].any() and not damage[:, :CHIP].any()


def test_windows_accept_integer_codes_and_reject_unknown_values(tmp_path):
    name = _write_chip(tmp_path / "post", X, Y, seed=1)
    _write_chip(tmp_path / "pre", X, Y, seed=2)
    square = _tile_center_square(X, Y)

    windows = windows_from_chips(tmp_path / "post", tmp_path / "pre", _labels([(square, 2)]), [name])
    assert set(np.unique(windows[0]["damage"])) == {0, 2}

    with pytest.raises(ValueError, match="unrecognised damage value"):
        windows_from_chips(tmp_path / "post", tmp_path / "pre", _labels([(square, "partial")]), [name])


def test_windows_fail_when_a_pre_chip_is_missing(tmp_path):
    name = _write_chip(tmp_path / "post", X, Y, seed=1)
    (tmp_path / "pre").mkdir()
    labels = _labels([(_tile_center_square(X, Y), "destroyed")])
    with pytest.raises(RasterioIOError):
        windows_from_chips(tmp_path / "post", tmp_path / "pre", labels, [name])


def test_area_inputs_mosaic_chips_fetch_pre_over_their_bounds_and_clip_buildings(tmp_path):
    for x in (X, X + 1):
        _write_chip(tmp_path / "chips", x, Y, seed=x)
    west, south, _, _ = mercantile.bounds(X, Y, ZOOM)
    _, _, east, north = mercantile.bounds(X + 1, Y, ZOOM)
    margin = 0.01
    pre_path = tmp_path / "pre_source.tif"
    with rasterio.open(
        pre_path,
        "w",
        driver="GTiff",
        width=400,
        height=400,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_bounds(west - margin, south - margin, east + margin, north + margin, 400, 400),
    ) as dst:
        dst.write(np.full((3, 400, 400), 120, dtype=np.uint8))
    buildings_path = tmp_path / "buildings.geojson"
    inside = _tile_center_square(X, Y)
    far = box(west + 1, south + 1, west + 1.001, south + 1.001)
    _labels([(inside, "inside"), (far, "far")]).to_file(buildings_path, driver="GeoJSON")

    post_raster, pre_raster, buildings = area_inputs(
        tmp_path / "chips", str(pre_path), str(buildings_path), tmp_path / "work"
    )

    with rasterio.open(post_raster) as post:
        assert (post.width, post.height) == (2 * CHIP, CHIP)
        post_bounds = post.bounds
    with rasterio.open(pre_raster) as pre:
        # The window is snapped to whole source pixels, so its edges sit within one pixel of the chips.
        pixel = pre.res[0]
        assert abs(pre.bounds.left - post_bounds.left) <= pixel
        assert abs(pre.bounds.right - post_bounds.right) <= pixel
    assert list(buildings["damage"]) == ["inside"]


@pytest.mark.slow
def test_windows_fine_tune_and_evaluate_with_the_damage_model(tmp_path):
    """Real fit on chip windows; downloads the damage checkpoint and the encoder."""
    from dda.pipeline.fewshot_damage import evaluate_damage_windows, fit_damage_windows

    names = []
    rows = []
    for block in range(8):
        x = X + 2 * block
        names.append(_write_chip(tmp_path / "post", x, Y, seed=block))
        _write_chip(tmp_path / "pre", x, Y, seed=100 + block)
        rows.append((_tile_center_square(x, Y), ("no-damage", "destroyed")[block % 2]))
    windows = windows_from_chips(tmp_path / "post", tmp_path / "pre", _labels(rows), names)
    assert len(windows) == 8

    result = fit_damage_windows(windows[:4], windows[4:], tmp_path / "out", epochs=1, batch_size=2)
    metrics = evaluate_damage_windows(windows[4:], result.best_ckpt)

    assert result.best_ckpt.exists()
    assert set(metrics) == {"dmg_macro_f1", "dmg_f1", "loc_iou"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
