import functools
import logging
import os
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from dda.config import DINOV3_MEAN, DINOV3_STD, N_DAMAGE_CLASSES
from dda.hf_utils import resolve_sha_pinned_ckpt
from dda.model import DinoV3DamageLit
from dda.pool import assign_damage, mark_uncovered_as_no_data
from dda.tiling import blend_window, gaussian_kernel, imaged_mask, pad_tile, read_aligned_pair, window_origins

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
def _mean_std(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(DINOV3_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(DINOV3_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def _batch_normalize(tiles_uint8: np.ndarray, device: str) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 3, H, W) float32 normalised, on device."""
    t = torch.from_numpy(tiles_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean, std = _mean_std(device)
    return (t - mean) / std


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
    kernel = gaussian_kernel(window)

    coords = window_origins(height, width, window, stride)
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
            blend_window(prob_acc, weight_acc, probs[i], r, c, kernel)

    with torch.inference_mode():
        batch_post: list[np.ndarray] = []
        batch_pre: list[np.ndarray] | None = [] if pre is not None else None
        batch_coords: list[tuple[int, int]] = []
        for r, c in coords:
            batch_post.append(pad_tile(post[r : r + window, c : c + window, :], window))
            if pre is not None and batch_pre is not None:
                batch_pre.append(pad_tile(pre[r : r + window, c : c + window, :], window))
            batch_coords.append((r, c))
            if len(batch_coords) >= INFER_BATCH_SIZE:
                flush(batch_post, batch_pre, batch_coords)
                batch_post, batch_coords = [], []
                if batch_pre is not None:
                    batch_pre = []
        if batch_coords:
            flush(batch_post, batch_pre, batch_coords)

    return prob_acc / weight_acc


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
    gdf = mark_uncovered_as_no_data(gdf, imaged_mask(post, pre), transform)

    from dda.pipeline.geowrite import write_dual

    out_geojson = Path(out_geojson)
    write_dual(gdf, out_geojson)
    log.info("Wrote %d buildings with damage -> %s(.geojson|.parquet)", len(gdf), out_geojson.with_suffix(""))
    return out_geojson
