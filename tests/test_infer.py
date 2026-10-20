import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from dda.tiling import coregister, radiometric_normalize


def test_stretch_expands_compressed_range_to_full_scale():
    # a low-contrast scene compressed into [100, 140]
    rng = np.random.default_rng(0)
    image = rng.integers(100, 141, size=(64, 64, 3), dtype=np.uint8)
    out = radiometric_normalize(image)
    assert out.dtype == np.uint8
    assert out.min() == 1  # zero is reserved for pixels outside the imagery
    assert out.max() == 255
    assert out.std() > image.std()  # contrast increased


def test_black_border_pixels_excluded_and_kept_black():
    rng = np.random.default_rng(1)
    image = rng.integers(80, 161, size=(32, 32, 3), dtype=np.uint8)
    image[:8, :] = 0  # nodata border across all channels
    out = radiometric_normalize(image)
    assert (out[:8, :] == 0).all()  # border stays black, not stretched
    assert out[8:].max() == 255  # valid interior is stretched to full scale


def test_imaged_pixels_darker_than_the_low_percentile_stay_non_zero():
    rng = np.random.default_rng(2)
    image = rng.integers(80, 161, size=(64, 64, 3), dtype=np.uint8)
    image[10:20, 10:20] = 5  # deep shadow, below p2 in every channel
    out = radiometric_normalize(image)
    assert out[10:20, 10:20].any(axis=2).all()


def test_rejects_non_rgb_shape():
    with pytest.raises(ValueError, match="HxWx3"):
        radiometric_normalize(np.zeros((10, 10), dtype=np.uint8))


def test_rejects_all_zero_image():
    with pytest.raises(ValueError, match="no valid"):
        radiometric_normalize(np.zeros((10, 10, 3), dtype=np.uint8))


def test_rejects_degenerate_dynamic_range():
    with pytest.raises(ValueError, match="degenerate dynamic range"):
        radiometric_normalize(np.full((16, 16, 3), 50, dtype=np.uint8))


def test_coregister_reprojects_pre_onto_post_grid():
    profile = dict(
        driver="GTiff",
        height=20,
        width=20,
        count=3,
        dtype="uint8",
        crs="EPSG:32619",
        transform=from_origin(0, 20, 1, 1),
    )
    with MemoryFile() as mf:
        with mf.open(**profile) as ds:
            ds.write(np.full((3, 20, 20), 120, dtype=np.uint8))
        with mf.open() as ds:
            out = coregister(ds, (10, 10), from_origin(0, 20, 2, 2), ds.crs)
    assert out.shape == (10, 10, 3)  # resampled onto the post grid
    assert out.dtype == np.uint8
    assert out.max() > 0  # overlapping area was reprojected, not left black
