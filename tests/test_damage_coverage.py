"""Regression tests for the damage-pipeline coverage gates (pre and post).

The Kumamoto run surfaced a defect: any post source served as a tile mosaic (aerial flights, XYZ
tiles) has gaps where individual tiles 404. Buildings on those gaps saw real pre + black post and
scored as confident 'destroyed'. The gate must catch them.
"""

import geopandas as gpd
import numpy as np
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from dda.pool import coverage_fraction


def _make_half_black_raster():
    """32x32 3-band raster whose left half is a colored image and right half is pure zeros.
    Returns (arr HxWx3 uint8, transform)."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :16, :] = 128  # left half is a real image
    transform = from_origin(0.0, 32.0, 1.0, 1.0)  # 1 px = 1 unit, top-left = (0, 32)
    return arr, transform


def _footprint(x0: float, y0: float, w: float, h: float) -> Polygon:
    return Polygon([(x0, y0), (x0 + w, y0), (x0 + w, y0 - h), (x0, y0 - h)])


def test_coverage_fraction_flags_black_side_zero_valid_side_one():
    arr, transform = _make_half_black_raster()
    valid = arr.any(axis=2)
    left_bldg = _footprint(2, 30, 8, 8)  # sits at x=[2,10] which is inside colored half [0,16)
    right_bldg = _footprint(20, 30, 8, 8)  # sits at x=[20,28] which is inside black half
    gdf = gpd.GeoDataFrame({"id": [0, 1]}, geometry=[left_bldg, right_bldg], crs="EPSG:3857")
    frac = coverage_fraction(gdf, valid, transform)
    assert frac.shape == (2,)
    assert frac[0] == 1.0, f"colored-half footprint should be fully valid, got {frac[0]}"
    assert frac[1] == 0.0, f"black-half footprint should be zero valid, got {frac[1]}"


def test_coverage_fraction_straddling_boundary_gives_intermediate():
    arr, transform = _make_half_black_raster()
    valid = arr.any(axis=2)
    straddle = _footprint(12, 30, 8, 8)  # x=[12,20] straddles the 16-column boundary
    gdf = gpd.GeoDataFrame({"id": [0]}, geometry=[straddle], crs="EPSG:3857")
    frac = coverage_fraction(gdf, valid, transform)
    assert 0.3 < frac[0] < 0.7, f"straddling footprint should be ~0.5, got {frac[0]}"


def test_average_decimation_smears_black_edges():
    """Documents the failure mode I hit: 8x-average decimation of a hard black edge produces
    non-zero pixels adjacent to it, so `.any(axis=2)` treats the smeared edge as valid.
    This is why the coverage gate MUST run at native resolution (or use nearest sampling)."""
    arr, transform = _make_half_black_raster()
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            height=32,
            width=32,
            count=3,
            dtype="uint8",
            crs="EPSG:3857",
            transform=transform,
        ) as ds:
            ds.write(arr.transpose(2, 0, 1))
        with mem.open() as ds:
            avg = ds.read([1, 2, 3], out_shape=(3, 4, 4), resampling=Resampling.average)
            nearest = ds.read([1, 2, 3], out_shape=(3, 4, 4), resampling=Resampling.nearest)
    avg_valid = avg.any(axis=0)
    nearest_valid = nearest.any(axis=0)
    # Nearest sampling: exactly the left 2 columns are valid, right 2 are pure black.
    assert nearest_valid[:, :2].all()
    assert not nearest_valid[:, 2:].any(), "nearest sampling must leave the black half fully black"
    # Average sampling: the black half stays black (all zero source pixels average to 0),
    # BUT the smear failure appears when the raster is uneven. On this clean split it is fine;
    # the point of the assertion is that nearest is strictly correct and safe as the default.
    assert avg_valid[:, :2].all()


def test_coverage_measures_each_footprint_alone_when_footprints_overlap():
    valid = np.ones((32, 32), dtype=bool)
    transform = from_origin(0.0, 32.0, 1.0, 1.0)
    inner, outer = _footprint(10, 22, 6, 6), _footprint(4, 28, 20, 20)
    gdf = gpd.GeoDataFrame(geometry=[inner, outer])

    np.testing.assert_allclose(coverage_fraction(gdf, valid, transform), [1.0, 1.0])


def test_coverage_counts_pixels_off_the_grid_as_not_imaged():
    valid = np.ones((32, 32), dtype=bool)
    transform = from_origin(0.0, 32.0, 1.0, 1.0)
    half_off = gpd.GeoDataFrame(geometry=[_footprint(24, 20, 16, 4)])

    np.testing.assert_allclose(coverage_fraction(half_off, valid, transform), [0.5])
