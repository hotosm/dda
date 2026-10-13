import functools
import logging
import os
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import rasterio
import torch
from huggingface_hub import hf_hub_download
from rasterio.warp import Resampling, reproject
from scipy.signal.windows import gaussian as gauss1d
from scipy.special import softmax

from dda.config import N_DAMAGE_CLASSES
from dda.data import DINOV3_MEAN, DINOV3_STD
from dda.hf_utils import resolve_sha_pinned_ckpt
from dda.model import DinoV3DamageLit
from dda.pool import assign_damage

log = logging.getLogger(__name__)

DAMAGE_CKPT_SHA256 = "87485943ee286973168314a31ab62c0618e60d9b13d2ae7942dc82a4cee3eab7"
INFER_BATCH_SIZE = int(os.environ.get("DDA_INFER_BATCH", "16"))
INFER_COMPILE = os.environ.get("DDA_INFER_COMPILE", "1") != "0"


def resolve_ckpt(cfg, ckpt_path: str | Path | None = None) -> str:
    return str(
        resolve_sha_pinned_ckpt(
            cfg.model_repo,
            cfg.model_file,
            DAMAGE_CKPT_SHA256,
            override_path=ckpt_path,
            label="damage ckpt",
        )
    )


def load_model(ckpt_path: str | Path, cfg, device: str = "cuda") -> DinoV3DamageLit:
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    # weights_only=False: our own checkpoint, whose hparams carry non-tensor config objects.
    model = DinoV3DamageLit.load_from_checkpoint(
        str(ckpt_path), map_location=device, ckpt_path=encoder_ckpt, weights_only=False
    )
    model.eval()
    if INFER_COMPILE and device.startswith("cuda"):
        # `reduce-overhead` uses CUDA Graphs; pair with padded fixed-size batches in
        # sliding_window_prob or the graph recompiles on any shape drift.
        # The compiled wrapper forwards attribute access to the module it wraps.
        model = cast(DinoV3DamageLit, torch.compile(model, mode="reduce-overhead", fullgraph=False))
        log.info("torch.compile: reduce-overhead")
    return model


@functools.cache
def _gaussian_kernel(size: int, sigma_frac: float = 0.125) -> np.ndarray:
    w = gauss1d(size, std=sigma_frac * size)
    k = np.outer(w, w)
    return k / k.max()


@functools.cache
def _mean_std(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(DINOV3_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINOV3_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def _batch_normalize(tiles_uint8: np.ndarray, device: str) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float32 normalised, on device."""
    t = torch.from_numpy(tiles_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean, std = _mean_std(device)
    return (t - mean) / std


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
        out[..., channel_index] = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return out


def _coregister(pre_src, out_hw: tuple[int, int], dst_transform, dst_crs) -> np.ndarray:
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


def _pad(tile: np.ndarray, window: int) -> np.ndarray:
    if tile.shape[0] == window and tile.shape[1] == window:
        return tile
    pad = np.zeros((window, window, 3), dtype=np.uint8)
    pad[: tile.shape[0], : tile.shape[1]] = tile
    return pad


def _window_origins(height: int, width: int, window: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners of the sliding windows; the last row and column are pinned to the image edge."""
    rows = list(range(0, max(1, height - window + 1), stride))
    cols = list(range(0, max(1, width - window + 1), stride))
    if rows[-1] + window < height:
        rows.append(height - window)
    if cols[-1] + window < width:
        cols.append(width - window)
    return [(r, c) for r in rows for c in cols]


def sliding_window_prob(
    model: DinoV3DamageLit,
    post: np.ndarray,
    pre: np.ndarray | None,
    window: int,
    stride: int,
    temperature: float,
    device: str,
) -> np.ndarray:
    """Gaussian-blended per-class damage probabilities, shape (C, H, W). Tiles are batched
    to `INFER_BATCH_SIZE` per forward pass, and the forward runs under fp16 autocast on CUDA."""
    height, width = post.shape[:2]
    prob_acc = np.zeros((N_DAMAGE_CLASSES, height, width), dtype=np.float32)
    weight_acc = np.zeros((height, width), dtype=np.float32)
    kernel = _gaussian_kernel(window)

    coords = _window_origins(height, width, window, stride)
    on_cuda = device.startswith("cuda")
    zero_tile = np.zeros((window, window, 3), dtype=np.uint8)

    def flush(
        batch_post: list[np.ndarray],
        batch_pre: list[np.ndarray] | None,
        batch_coords: list[tuple[int, int]],
    ) -> None:
        # Pad to INFER_BATCH_SIZE so torch.compile's CUDA-Graphs mode never re-records.
        n_real = len(batch_coords)
        while len(batch_post) < INFER_BATCH_SIZE:
            batch_post.append(zero_tile)
            if batch_pre is not None:
                batch_pre.append(zero_tile)
        x_post = _batch_normalize(np.stack(batch_post, axis=0), device)
        x_pre = _batch_normalize(np.stack(batch_pre, axis=0), device) if batch_pre is not None else None
        if on_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, dmg_logits = model(x_post, x_pre)
        else:
            _, dmg_logits = model(x_post, x_pre)
        probs = torch.softmax(dmg_logits.float() / temperature, dim=1).cpu().numpy()
        for i in range(n_real):
            r, c = batch_coords[i]
            h = min(window, height - r)
            w = min(window, width - c)
            prob_acc[:, r : r + h, c : c + w] += probs[i, :, :h, :w] * kernel[:h, :w]
            weight_acc[r : r + h, c : c + w] += kernel[:h, :w]

    with torch.inference_mode():
        batch_post: list[np.ndarray] = []
        batch_pre: list[np.ndarray] | None = [] if pre is not None else None
        batch_coords: list[tuple[int, int]] = []
        for r, c in coords:
            batch_post.append(_pad(post[r : r + window, c : c + window, :], window))
            if pre is not None and batch_pre is not None:
                batch_pre.append(_pad(pre[r : r + window, c : c + window, :], window))
            batch_coords.append((r, c))
            if len(batch_coords) >= INFER_BATCH_SIZE:
                flush(batch_post, batch_pre, batch_coords)
                batch_post, batch_coords = [], []
                if batch_pre is not None:
                    batch_pre = []
        if batch_coords:
            flush(batch_post, batch_pre, batch_coords)

    return prob_acc / weight_acc


def sliding_window_prob_onnx(
    session: ort.InferenceSession,
    post: np.ndarray,
    pre: np.ndarray,
    window: int,
    stride: int,
    temperature: float,
) -> np.ndarray:
    """`sliding_window_prob` for the ONNX export of `dda.export.export_onnx`, one window per run."""
    height, width = post.shape[:2]
    prob_acc = np.zeros((N_DAMAGE_CLASSES, height, width), dtype=np.float32)
    weight_acc = np.zeros((height, width), dtype=np.float32)
    kernel = _gaussian_kernel(window)
    mean = np.asarray(DINOV3_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(DINOV3_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    def normalize(tile: np.ndarray) -> np.ndarray:
        chw = _pad(tile, window).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
        return (chw - mean) / std

    for r, c in _window_origins(height, width, window, stride):
        inputs = {
            "post": normalize(post[r : r + window, c : c + window, :]),
            "pre": normalize(pre[r : r + window, c : c + window, :]),
        }
        logits = np.asarray(session.run(["damage_logits"], inputs)[0])
        probs = softmax(logits[0] / temperature, axis=0)
        h = min(window, height - r)
        w = min(window, width - c)
        prob_acc[:, r : r + h, c : c + w] += probs[:, :h, :w] * kernel[:h, :w]
        weight_acc[r : r + h, c : c + w] += kernel[:h, :w]
    return prob_acc / weight_acc


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
            pre = _coregister(src, post.shape[:2], transform, crs)

    if cfg.radiometric_normalize:
        post = radiometric_normalize(post)
        if pre is not None:
            pre = radiometric_normalize(pre)
    return post, pre, transform, crs


def predict_damage_onnx(
    session: ort.InferenceSession,
    post_raster: str | Path,
    pre_raster: str | Path,
    buildings: gpd.GeoDataFrame,
    cfg,
) -> gpd.GeoDataFrame:
    """Damage class and confidence for each building from the ONNX export of `dda.export.export_onnx`."""
    post, pre, transform, crs = read_aligned_pair(cfg, post_raster, pre_raster)
    assert pre is not None
    prob = sliding_window_prob_onnx(session, post, pre, cfg.tile_window, cfg.tile_stride, cfg.temperature)
    return assign_damage(prob, transform, crs, buildings, pool_op=cfg.pool_op, percentile=cfg.pool_percentile)


def predict_damage(
    cfg,
    ckpt_path: str | Path,
    post_raster: str | Path,
    buildings_geojson: str | Path,
    out_geojson: str | Path,
    pre_raster: str | Path | None = None,
    device: str | None = None,
) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, cfg, device=device)
    post, pre, transform, crs = read_aligned_pair(cfg, post_raster, pre_raster)

    prob = sliding_window_prob(model, post, pre, cfg.tile_window, cfg.tile_stride, cfg.temperature, device)

    buildings = gpd.read_file(buildings_geojson)
    gdf = assign_damage(
        prob,
        transform,
        crs,
        buildings,
        pool_op=cfg.pool_op,
        percentile=cfg.pool_percentile,
    )

    from dda.pipeline.geowrite import write_dual

    out_geojson = Path(out_geojson)
    write_dual(gdf, out_geojson)
    log.info("Wrote %d buildings with damage -> %s(.geojson|.parquet)", len(gdf), out_geojson.with_suffix(""))
    return out_geojson
