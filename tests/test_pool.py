import geopandas as gpd
import numpy as np
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

from dda.config import N_DAMAGE_CLASSES
from dda.pool import _pool_one, assign_damage, filter_by_confidence, parse_confidence_threshold


def _onehot_pixels(class_counts: dict[int, int]) -> np.ndarray:
    cols = []
    for cls, n in class_counts.items():
        col = np.zeros((N_DAMAGE_CLASSES, n))
        col[cls] = 1.0
        cols.append(col)
    return np.concatenate(cols, axis=1)


def test_percentile_keeps_local_damage_that_mean_dilutes():
    pix = _onehot_pixels({0: 5, 3: 5})  # half intact, half destroyed
    cls_pct = _pool_one(pix, "percentile", 80.0)
    cls_mean = _pool_one(pix, "mean", 80.0)
    assert cls_pct > cls_mean


def test_uniform_region_class():
    pix = _onehot_pixels({2: 20})
    assert _pool_one(pix, "percentile", 80.0) == 2


def test_assign_damage_uniform_and_nodata():
    height = width = 8
    transform = from_origin(0, height, 1, 1)
    prob = np.zeros((N_DAMAGE_CLASSES, height, width), dtype=np.float32)
    prob[3, 1:4, 1:4] = 1.0  # destroyed block
    prob[0] = np.where(prob.sum(axis=0) == 0, 1.0, 0.0)  # rest no-damage

    inside = box(1.2, height - 3.8, 3.8, height - 1.2)  # over the destroyed block
    outside = box(100, 100, 101, 101)  # off-raster -> no-data
    buildings = gpd.GeoDataFrame(geometry=[inside, outside], crs="EPSG:3857")

    out = assign_damage(prob, transform, "EPSG:3857", buildings, pool_op="percentile", percentile=80.0)
    assert out.iloc[0]["damage"] == "destroyed"
    assert out.iloc[0]["damage_confidence"] > 0.9
    assert out.iloc[1]["damage_class"] == -1
    assert out.iloc[1]["damage"] == "no-data"
    assert np.isnan(out.iloc[1]["damage_confidence"])
    assert set(out.columns) == {"damage_class", "damage", "damage_confidence", "geometry"}


def _assessed(confidences: list[float]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"damage_confidence": confidences},
        geometry=[box(i, 0, i + 1, 1) for i in range(len(confidences))],
        crs="EPSG:4326",
    )


def test_confidence_filter_drops_low_confidence_and_keeps_no_data():
    buildings = _assessed([0.2, 0.5, 0.9, float("nan")])

    kept = filter_by_confidence(buildings, 0.5)

    assert kept["damage_confidence"].tolist()[:2] == [0.5, 0.9]
    assert np.isnan(kept["damage_confidence"].iloc[2])
    assert len(kept) == 3


def test_zero_confidence_threshold_keeps_every_building():
    buildings = _assessed([0.0, 0.3, float("nan")])

    assert len(filter_by_confidence(buildings, 0.0)) == 3


@pytest.mark.parametrize("value", [1.5, -0.1, "abc", None, [0.5], True, float("nan")])
def test_invalid_confidence_threshold_is_rejected(value):
    with pytest.raises(ValueError, match="confidence_threshold"):
        parse_confidence_threshold(value)


@pytest.mark.parametrize(("value", "expected"), [(0, 0.0), (0.5, 0.5), (1, 1.0)])
def test_valid_confidence_threshold_is_parsed(value, expected):
    assert parse_confidence_threshold(value) == expected
