"""Damage inference with onnxruntime over the ONNX export of `dda.export.export_onnx`."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import onnxruntime as ort
from scipy.special import softmax

from dda.config import DINOV3_MEAN, DINOV3_STD, N_DAMAGE_CLASSES
from dda.pool import assign_damage, mark_uncovered_as_no_data
from dda.tiling import blend_window, gaussian_kernel, pad_tile, read_aligned_pair, window_origins


def sliding_window_prob_onnx(
    session: ort.InferenceSession,
    post: np.ndarray,
    pre: np.ndarray,
    window: int,
    stride: int,
    temperature: float,
) -> np.ndarray:
    """`dda.infer.sliding_window_prob` for the ONNX export, one window per run."""
    height, width = post.shape[:2]
    probability_sum = np.zeros((N_DAMAGE_CLASSES, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    kernel = gaussian_kernel(window)
    mean = np.asarray(DINOV3_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(DINOV3_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def normalize(tile: np.ndarray) -> np.ndarray:
        scaled_tile = pad_tile(tile, window).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return (scaled_tile - mean) / std

    for row, col in window_origins(height, width, window, stride):
        inputs = {
            "post": normalize(post[row : row + window, col : col + window, :]),
            "pre": normalize(pre[row : row + window, col : col + window, :]),
        }
        logits = np.asarray(session.run(["damage_logits"], inputs)[0])
        blend_window(probability_sum, weight_sum, softmax(logits[0] / temperature, axis=0), row, col, kernel)
    return probability_sum / weight_sum


def predict_damage_onnx(
    session: ort.InferenceSession,
    post_raster: str | Path,
    pre_raster: str | Path,
    buildings: gpd.GeoDataFrame,
    cfg,
) -> gpd.GeoDataFrame:
    """Damage class and confidence per building; buildings under `MIN_COVERAGE_FRACTION` of pre and post
    imagery are no-data."""
    post, pre, transform, crs = read_aligned_pair(cfg, post_raster, pre_raster)
    if pre is None:
        raise ValueError("predict_damage_onnx needs a pre-event raster")
    prob = sliding_window_prob_onnx(session, post, pre, cfg.tile_window, cfg.tile_stride, cfg.temperature)
    scored = assign_damage(
        prob, transform, crs, buildings, pool_op=cfg.pool_op, percentile=cfg.pool_percentile
    )
    return mark_uncovered_as_no_data(scored, post.any(axis=2) & pre.any(axis=2), transform)
