"""Reproject pre onto post grid, measure any residual drift (SIFT), write both as COGs.

Dynamic-range normalisation is the damage stage's job: `dda.pipeline.damage._process_block`
runs `radiometric_normalize` (2-98 percentile stretch) per scene before feeding the model,
so this stage stops at spatial alignment. Viewer-matched tone is titiler `color_formula`
at render time.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import odc.geo.xr  # noqa: F401  # registers .odc accessor on xarray
import rasterio
import rioxarray  # registers .rio accessor on xarray
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject
from PIL import Image
from rasterio.enums import Resampling

log = logging.getLogger(__name__)

CHECKERBOARD_BLOCKS = 12
SIFT_DECIMATION = 8
SIFT_LOWE_RATIO = 0.75
SIFT_RANSAC_REPROJ_PX = 5.0
SIFT_MIN_INLIERS = 10

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
    "GDAL_CACHEMAX": "200",
    "AWS_NO_SIGN_REQUEST": "YES",
}


@dataclass
class DriftResult:
    dy_px: float
    dx_px: float
    pixel_size_m: float
    magnitude_m: float
    sift_matches: int
    sift_inliers: int


def coregister(
    pre_raw: Path,
    post_raw: Path,
    pre_aligned: Path,
    post_aligned: Path,
    drift_json: Path,
    check_png: Path,
    *,
    keep_raw: bool = False,
    skip_drift: bool = False,
) -> DriftResult:
    """Reproject `pre_raw` onto `post_raw`'s grid; optionally measure sub-pixel drift and shift
    the pre to match; write both as COGs; render a pre/post checkerboard for QA."""
    # os.environ carries into rasterio's in-process reads and into dask.distributed
    # LocalCluster subprocess workers (they inherit env). A remote cluster would need
    # per-worker configuration instead.
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)
    pre_aligned.parent.mkdir(parents=True, exist_ok=True)
    drift_json.parent.mkdir(parents=True, exist_ok=True)

    # Post stays on its own grid; write as COG untouched.
    _write_cog_from_source(post_raw, post_aligned)

    # Pre is lazily reprojected onto post's geobox via odc-geo (dask-parallel).
    _reproject_pre_to_post_grid(pre_raw, post_aligned, pre_aligned)

    drift = _identity_drift(post_aligned) if skip_drift else _measure_and_shift(pre_aligned, post_aligned)

    _render_checkerboard(pre_aligned, post_aligned, check_png)
    if not keep_raw:
        for p in (pre_raw, post_raw):
            if p.exists():
                p.unlink()
        log.info("coregister: deleted raw pre.tif + post.tif (set keep_raw=True to preserve)")

    drift_json.write_text(json.dumps(asdict(drift), indent=2))
    log.info(
        "drift dy=%.2f px dx=%.2f px (%.1f m at %.2f m/px)",
        drift.dy_px,
        drift.dx_px,
        drift.magnitude_m,
        drift.pixel_size_m,
    )
    return drift


def _write_cog_from_source(src: Path, dst: Path) -> None:
    """One-pass COG write via GDAL's COG driver; keeps the source's grid + values byte-for-byte."""
    from rasterio.shutil import copy as rio_copy

    if dst.exists():
        dst.unlink()
    rio_copy(
        str(src),
        str(dst),
        driver="COG",
        compress="DEFLATE",
        blocksize=512,
        overview_resampling="average",
        num_threads="ALL_CPUS",
        bigtiff="YES",
    )
    with rasterio.open(dst) as s:
        log.info(
            "cog: %s -> %d overview levels, %.2f GB", dst.name, len(s.overviews(1)), dst.stat().st_size / 1e9
        )


def _reproject_pre_to_post_grid(pre_raw: Path, post_aligned: Path, out: Path) -> None:
    """Lazy warp of pre onto post's geobox using odc-geo; writes result as a single-pass COG.
    `xr_reproject` returns a dask-backed array (verified: no RAM growth at graph-build time),
    and `.rio.to_raster` iterates its blocks; peak memory scales with chunk size, not raster size."""
    with rasterio.open(post_aligned) as ps:
        target_geobox = GeoBox.from_rio(ps)
    src = rioxarray.open_rasterio(pre_raw, chunks={"x": 2048, "y": 2048}, band_as_variable=False)
    if not isinstance(src, xr.DataArray):
        raise TypeError(f"expected a single raster band stack from {pre_raw}, got {type(src).__name__}")
    reprojected = xr_reproject(src, target_geobox, resampling="bilinear", dst_nodata=0)
    reprojected = reprojected.astype("uint8")
    reprojected.rio.to_raster(
        out,
        driver="COG",
        dtype="uint8",
        compress="DEFLATE",
        bigtiff="YES",
        overview_resampling="average",
        num_threads="ALL_CPUS",
    )
    with rasterio.open(out) as s:
        log.info(
            "cog: %s -> %d overview levels, %.2f GB", out.name, len(s.overviews(1)), out.stat().st_size / 1e9
        )


def _identity_drift(post_aligned: Path) -> DriftResult:
    log.info("coregister: skip_drift=True; identity homography (no SIFT)")
    with rasterio.open(post_aligned) as s:
        px_m = _mean_pixel_size_m(s)
    return DriftResult(
        dy_px=0.0, dx_px=0.0, pixel_size_m=px_m, magnitude_m=0.0, sift_matches=0, sift_inliers=0
    )


def _measure_and_shift(pre_aligned: Path, post_aligned: Path) -> DriftResult:
    drift = _measure_homography(pre_aligned, post_aligned)
    homog = np.array([[1, 0, -drift.dx_px], [0, 1, -drift.dy_px], [0, 0, 1]], dtype=np.float64)
    _apply_homography(pre_aligned, homog)
    return drift


def _mean_pixel_size_m(src) -> float:
    if src.crs.is_projected:
        return float(abs(src.transform.a) + abs(src.transform.e)) / 2.0
    centre_lat = (src.bounds.top + src.bounds.bottom) / 2.0
    metres_per_deg = 111_320.0 * np.cos(np.deg2rad(centre_lat))
    return float((abs(src.transform.a) + abs(src.transform.e)) / 2.0 * metres_per_deg)


def _decimated_grayscale(src, step: int) -> np.ndarray:
    arr = src.read(
        [1, 2, 3], out_shape=(3, src.height // step, src.width // step), resampling=Resampling.average
    ).astype(np.float32)
    return arr.mean(axis=0)


def _measure_homography(pre_aligned: Path, post: Path) -> DriftResult:
    """SIFT+RANSAC on decimated grayscale overviews; returns translation in full-res pixels."""
    step = SIFT_DECIMATION
    with rasterio.open(post) as ps, rasterio.open(pre_aligned) as pr:
        post_gray = _decimated_grayscale(ps, step).astype(np.uint8)
        pre_gray = _decimated_grayscale(pr, step).astype(np.uint8)
        pixel_size_m = _mean_pixel_size_m(ps)
    h = min(post_gray.shape[0], pre_gray.shape[0])
    w = min(post_gray.shape[1], pre_gray.shape[1])
    post_gray, pre_gray = post_gray[:h, :w], pre_gray[:h, :w]
    post_norm = cv2.normalize(post_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # ty: ignore[no-matching-overload]
    pre_norm = cv2.normalize(pre_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # ty: ignore[no-matching-overload]

    sift = cv2.SIFT_create()  # ty: ignore[unresolved-attribute]
    post_kp, post_desc = sift.detectAndCompute(post_norm, None)
    pre_kp, pre_desc = sift.detectAndCompute(pre_norm, None)
    if pre_desc is None or post_desc is None:
        raise RuntimeError("SIFT found no descriptors; imagery may be featureless")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(pre_desc, post_desc, k=2)
    good = [m for m, n in matches if m.distance < SIFT_LOWE_RATIO * n.distance]
    if len(good) < SIFT_MIN_INLIERS:
        raise RuntimeError(f"only {len(good)} good matches (need >= {SIFT_MIN_INLIERS})")
    src_pts = np.array([pre_kp[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = np.array([post_kp[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    homog_dec, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, SIFT_RANSAC_REPROJ_PX)
    if homog_dec is None:
        raise RuntimeError("RANSAC could not fit a homography")
    inliers = int(mask.sum())
    if inliers < SIFT_MIN_INLIERS:
        raise RuntimeError(f"only {inliers} inliers after RANSAC (need >= {SIFT_MIN_INLIERS})")
    dx_px = float(homog_dec[0, 2]) * step
    dy_px = float(homog_dec[1, 2]) * step
    magnitude_m = float(np.hypot(dx_px, dy_px) * pixel_size_m)
    log.info(
        "sift: %d matches, %d inliers, translation (dx=%.2f, dy=%.2f) px = %.1f m",
        len(good),
        inliers,
        dx_px,
        dy_px,
        magnitude_m,
    )
    return DriftResult(
        dy_px=dy_px,
        dx_px=dx_px,
        pixel_size_m=pixel_size_m,
        magnitude_m=magnitude_m,
        sift_matches=len(good),
        sift_inliers=inliers,
    )


def _apply_homography(pre_aligned: Path, homography: np.ndarray) -> None:
    """Warp each band with the drift-correction homography and rewrite as a single-pass COG."""
    with rasterio.open(pre_aligned) as src:
        bands = np.stack(
            [
                cv2.warpPerspective(
                    src.read(b),
                    homography,
                    (src.width, src.height),
                    flags=cv2.INTER_LINEAR,
                    borderValue=0,
                )
                for b in range(1, src.count + 1)
            ]
        )
        crs, transform = src.crs, src.transform
    da = xr.DataArray(
        bands,
        dims=("band", "y", "x"),
        coords={"band": np.arange(1, bands.shape[0] + 1)},
    )
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    cog_tmp = pre_aligned.with_suffix(pre_aligned.suffix + ".cog.tmp")
    da.rio.to_raster(
        cog_tmp,
        driver="COG",
        dtype="uint8",
        compress="DEFLATE",
        bigtiff="YES",
        overview_resampling="average",
        num_threads="ALL_CPUS",
    )
    cog_tmp.replace(pre_aligned)


def _render_checkerboard(
    pre_aligned: Path, post: Path, out_png: Path, blocks: int = CHECKERBOARD_BLOCKS
) -> None:
    """Alternating pre/post cells; roads and coastlines look continuous when alignment is clean."""
    with rasterio.open(post) as ps, rasterio.open(pre_aligned) as pr:
        h = min(ps.height, pr.height)
        w = min(ps.width, pr.width)
        step = 8
        post_arr = ps.read([1, 2, 3], out_shape=(3, h // step, w // step), resampling=Resampling.average)
        pre_arr = pr.read([1, 2, 3], out_shape=(3, h // step, w // step), resampling=Resampling.average)
    ch, cw = post_arr.shape[1], post_arr.shape[2]
    block_h = max(1, ch // blocks)
    block_w = max(1, cw // blocks)
    out = post_arr.copy()
    for by in range(blocks):
        for bx in range(blocks):
            if (by + bx) % 2 == 0:
                y0, y1 = by * block_h, (by + 1) * block_h
                x0, x1 = bx * block_w, (bx + 1) * block_w
                out[:, y0:y1, x0:x1] = pre_arr[:, y0:y1, x0:x1]
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.transpose(out, (1, 2, 0))).save(out_png)
    log.info("checkerboard: %s", out_png)
