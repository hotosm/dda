"""Raster alignment, stretching and sliding-window tiling shared by the torch and ONNX damage paths."""

import functools
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.signal.windows import gaussian as gauss1d


@functools.cache
def gaussian_kernel(size: int, sigma_frac: float = 0.125) -> np.ndarray:
    w = gauss1d(size, std=sigma_frac * size)
    k = np.outer(w, w)
    return k / k.max()


def radiometric_normalize(
    image: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> np.ndarray:
    """Per-channel percentile stretch of an RGB scene to the full 8-bit range; disaster imagery arrives
    uncalibrated with a compressed range that is out of distribution for the model's fixed normalization."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB image, got shape {image.shape}")
    valid = image.any(axis=2)
    if not valid.any():
        raise ValueError("image has no valid (non-zero) pixels to normalize")
    out = np.empty_like(image, dtype=np.uint8)
    for channel_index in range(3):
        channel = image[..., channel_index]
        low, high = np.percentile(channel[valid], [low_percentile, high_percentile])
        if high <= low:
            raise ValueError(
                f"degenerate dynamic range in channel {channel_index}: "
                f"p{low_percentile}={low:.1f} >= p{high_percentile}={high:.1f}"
            )
        scaled = (channel.astype(np.float32) - low) / (high - low) * 255.0
        # Imaged pixels stay at 1 or above, so zero keeps meaning "no imagery" after the stretch.
        out[..., channel_index] = np.where(valid, np.clip(scaled, 1.0, 255.0), 0).astype(np.uint8)
    return out


def coregister(pre_src, out_hw: tuple[int, int], dst_transform, dst_crs) -> np.ndarray:
    """Reproject the pre-event raster onto the post grid so pre and post share pixels for the change model."""
    height, width = out_hw
    out = np.zeros((3, height, width), dtype=np.uint8)
    for band in range(3):
        reproject(
            source=rasterio.band(pre_src, band + 1),
            destination=out[band],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.cubic,
        )
    return out.transpose(1, 2, 0)


def pad_tile(tile: np.ndarray, window: int) -> np.ndarray:
    if tile.shape[0] == window and tile.shape[1] == window:
        return tile
    pad = np.zeros((window, window, 3), dtype=np.uint8)
    pad[: tile.shape[0], : tile.shape[1]] = tile
    return pad


def window_origins(height: int, width: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of the sliding windows; the last row and column are pinned to the image edge."""
    rows = list(range(0, max(1, height - window + 1), stride))
    cols = list(range(0, max(1, width - window + 1), stride))
    if rows[-1] + window < height:
        rows.append(height - window)
    if cols[-1] + window < width:
        cols.append(width - window)
    return [(r, c) for r in rows for c in cols]


def blend_window(
    prob_acc: np.ndarray, weight_acc: np.ndarray, probs: np.ndarray, row: int, col: int, kernel: np.ndarray
) -> None:
    """Add one window's (C, window, window) probabilities, Gaussian-weighted and cropped at the image edge."""
    window = kernel.shape[0]
    height, width = weight_acc.shape
    h = min(window, height - row)
    w = min(window, width - col)
    prob_acc[:, row : row + h, col : col + w] += probs[:, :h, :w] * kernel[:h, :w]
    weight_acc[row : row + h, col : col + w] += kernel[:h, :w]


def read_aligned_pair(
    cfg, post_raster: str | Path, pre_raster: str | Path | None
) -> tuple[np.ndarray, np.ndarray | None, rasterio.Affine, rasterio.crs.CRS]:
    """Post and pre RGB as HxWx3 uint8 on the post grid, stretched when `cfg.radiometric_normalize`."""
    with rasterio.open(post_raster) as src:
        post = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        transform, crs = src.transform, src.crs

    pre = None
    if pre_raster is not None:
        with rasterio.open(pre_raster) as src:
            pre = coregister(src, post.shape[:2], transform, crs)

    if cfg.radiometric_normalize:
        post = radiometric_normalize(post)
        if pre is not None:
            pre = radiometric_normalize(pre)
    return post, pre, transform, crs
