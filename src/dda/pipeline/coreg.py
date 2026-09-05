"""Reproject pre to post grid, SIFT+RANSAC homography, photometric match, 2-98 stretch, COG, checkerboard."""

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.shutil import copy as rio_shutil_copy
from rasterio.warp import reproject

log = logging.getLogger(__name__)

DRIFT_DECIMATION = 8
CHECKERBOARD_BLOCKS = 12
STRETCH_PCT_LOW = 2.0
STRETCH_PCT_HIGH = 98.0
STRETCH_HIGH_MIN_SANITY = 128.0
SIFT_DECIMATION = 8
SIFT_LOWE_RATIO = 0.75
SIFT_RANSAC_REPROJ_PX = 5.0
SIFT_MIN_INLIERS = 10


@dataclass
class DriftResult:
    dy_px: float
    dx_px: float
    pixel_size_m: float
    magnitude_m: float
    homography: list[list[float]]
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
    calibrate_photometry: bool = True,
    stretch_percentiles: bool = True,
    keep_raw: bool = False,
    shift_direction: str = "pre_to_post",
) -> DriftResult:
    """Reproject pre, SIFT+RANSAC homography, optional photometric + stretch, write COGs and checkerboard.

    shift_direction=pre_to_post keeps the post frame authoritative (fAIr on Vantor);
    post_to_pre keeps the pre frame authoritative (OSM footprints traced against ESRI).
    """
    if shift_direction not in ("pre_to_post", "post_to_pre"):
        raise ValueError(f"shift_direction must be pre_to_post or post_to_pre, got {shift_direction!r}")
    pre_aligned.parent.mkdir(parents=True, exist_ok=True)
    drift_json.parent.mkdir(parents=True, exist_ok=True)

    _reproject_onto_post(pre_raw, post_raw, pre_aligned)
    drift = _measure_homography(pre_aligned, post_raw)
    homography = np.asarray(drift.homography, dtype=np.float64)
    shutil.copyfile(post_raw, post_aligned)
    if shift_direction == "pre_to_post":
        _apply_homography(pre_aligned, homography)
    else:
        _apply_homography(post_aligned, np.linalg.inv(homography))
    if calibrate_photometry:
        _apply_photometric_calibration(pre_aligned, post_aligned)
    if stretch_percentiles:
        _apply_shared_stretch_from_post(pre_aligned, post_aligned)
    _convert_to_cog(pre_aligned)
    _convert_to_cog(post_aligned)
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


def _gtiff_profile(height: int, width: int, crs, transform) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "yes",
    }


def _reproject_onto_post(pre_raw: Path, post: Path, out_aligned: Path) -> None:
    with rasterio.open(post) as post_src, rasterio.open(pre_raw) as pre_src:
        profile = _gtiff_profile(post_src.height, post_src.width, post_src.crs, post_src.transform)
        with rasterio.open(out_aligned, "w", **profile) as dst:
            for band in range(3):
                dest = np.zeros((post_src.height, post_src.width), dtype=np.uint8)
                reproject(
                    source=rasterio.band(pre_src, band + 1),
                    destination=dest,
                    dst_transform=post_src.transform,
                    dst_crs=post_src.crs,
                    resampling=Resampling.cubic,
                )
                dst.write(dest, band + 1)


def _decimated_grayscale(src: "rasterio.io.DatasetReader", step: int) -> np.ndarray:
    arr = src.read(
        [1, 2, 3],
        out_shape=(3, src.height // step, src.width // step),
        resampling=Resampling.average,
    ).astype(np.float32)
    return arr.mean(axis=0)


def _mean_pixel_size_m(src: "rasterio.io.DatasetReader") -> float:
    if src.crs.is_projected:
        return float(abs(src.transform.a) + abs(src.transform.e)) / 2.0
    centre_lat = (src.bounds.top + src.bounds.bottom) / 2.0
    metres_per_deg = 111_320.0 * np.cos(np.deg2rad(centre_lat))
    return float((abs(src.transform.a) + abs(src.transform.e)) / 2.0 * metres_per_deg)


def _measure_homography(pre_aligned: Path, post: Path) -> DriftResult:
    """SIFT+RANSAC homography on decimated overviews, rescaled to full-res."""
    with rasterio.open(post) as ps, rasterio.open(pre_aligned) as pr:
        step = SIFT_DECIMATION
        post_gray = _decimated_grayscale(ps, step).astype(np.uint8)
        pre_gray = _decimated_grayscale(pr, step).astype(np.uint8)
        pixel_size_m = _mean_pixel_size_m(ps)
    common = (
        min(post_gray.shape[0], pre_gray.shape[0]),
        min(post_gray.shape[1], pre_gray.shape[1]),
    )
    post_gray = post_gray[: common[0], : common[1]]
    pre_gray = pre_gray[: common[0], : common[1]]
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

    scale = np.diag([float(step), float(step), 1.0])
    scale_inv = np.diag([1.0 / step, 1.0 / step, 1.0])
    homog_full = scale @ homog_dec @ scale_inv

    dx_px = float(homog_full[0, 2])
    dy_px = float(homog_full[1, 2])
    magnitude_m = float(np.hypot(dx_px, dy_px) * pixel_size_m)
    log.info(
        "sift homography: %d matches, %d inliers, translation (dx=%.2f, dy=%.2f) px = %.1f m",
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
        homography=homog_full.tolist(),
        sift_matches=len(good),
        sift_inliers=inliers,
    )


def _apply_homography(pre_aligned: Path, homography: np.ndarray) -> None:
    """Warp each band with a full-res 3x3 homography and overwrite `pre_aligned`."""
    tmp = pre_aligned.with_suffix(pre_aligned.suffix + ".warped.tmp")
    with rasterio.open(pre_aligned) as src:
        profile = _gtiff_profile(src.height, src.width, src.crs, src.transform)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band in range(1, 4):
                arr = src.read(band)
                warped = cv2.warpPerspective(
                    arr,
                    homography,
                    (src.width, src.height),
                    flags=cv2.INTER_LINEAR,
                    borderValue=0,
                )
                dst.write(warped, band)
                del arr, warped
    tmp.replace(pre_aligned)


def _apply_photometric_calibration(pre_aligned: Path, post: Path) -> None:
    """Rescale pre per-band mean+std to match post; nodata (any-channel == 0) preserved as 0."""
    with rasterio.open(pre_aligned) as pre, rasterio.open(post) as ps:
        step = DRIFT_DECIMATION
        pre_dec = pre.read(
            [1, 2, 3],
            out_shape=(3, pre.height // step, pre.width // step),
            resampling=Resampling.average,
        )
        post_dec = ps.read(
            [1, 2, 3],
            out_shape=(3, ps.height // step, ps.width // step),
            resampling=Resampling.average,
        )
    common = (
        min(pre_dec.shape[1], post_dec.shape[1]),
        min(pre_dec.shape[2], post_dec.shape[2]),
    )
    pre_dec = pre_dec[:, : common[0], : common[1]]
    post_dec = post_dec[:, : common[0], : common[1]]
    valid = (pre_dec.sum(axis=0) > 0) & (post_dec.sum(axis=0) > 0)
    if valid.sum() < 1000:
        log.warning(
            "photometric calibration skipped: only %d overlapping valid pixels (need >=1000)",
            int(valid.sum()),
        )
        return

    means_pre, stds_pre, means_post, stds_post = [], [], [], []
    for b in range(3):
        p = pre_dec[b][valid].astype(np.float32)
        q = post_dec[b][valid].astype(np.float32)
        p_std, q_std = float(p.std()), float(q.std())
        if p_std == 0 or q_std == 0:
            log.warning(
                "photometric calibration: band %d has zero std (pre=%.3f, post=%.3f); "
                "offset-only for this band",
                b,
                p_std,
                q_std,
            )
            p_std = p_std or 1.0
            q_std = q_std or p_std
        means_pre.append(float(p.mean()))
        stds_pre.append(p_std)
        means_post.append(float(q.mean()))
        stds_post.append(q_std)
    log.info(
        "photometric calibration: pre RGB mean=%s std=%s -> post mean=%s std=%s",
        [round(m, 1) for m in means_pre],
        [round(s, 1) for s in stds_pre],
        [round(m, 1) for m in means_post],
        [round(s, 1) for s in stds_post],
    )

    tmp = pre_aligned.with_suffix(pre_aligned.suffix + ".calib.tmp")
    with rasterio.open(pre_aligned) as src:
        profile = _gtiff_profile(src.height, src.width, src.crs, src.transform)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band in range(1, 4):
                arr = src.read(band).astype(np.float32)
                nodata = arr == 0
                scale = stds_post[band - 1] / stds_pre[band - 1]
                out = (arr - means_pre[band - 1]) * scale + means_post[band - 1]
                out = np.clip(out, 0, 255).astype(np.uint8)
                out[nodata] = 0
                dst.write(out, band)
                del arr, out, nodata
    tmp.replace(pre_aligned)


def _compute_stretch_bounds(
    path: Path,
    label: str,
) -> tuple[list[float], list[float]] | None:
    """Per-band 2-98 bounds on valid pixels; None if mostly no-data, raises when all highs are too low."""
    with rasterio.open(path) as src:
        step = DRIFT_DECIMATION
        dec = src.read(
            [1, 2, 3],
            out_shape=(3, src.height // step, src.width // step),
            resampling=Resampling.average,
        )
    valid = dec.sum(axis=0) > 0
    if valid.sum() < 1000:
        log.warning(
            "%s stretch skipped: only %d valid pixels in decimated view (need >=1000)",
            label,
            int(valid.sum()),
        )
        return None
    lows: list[float] = []
    highs: list[float] = []
    for b in range(3):
        pixels = dec[b][valid].astype(np.float32)
        lo, hi = np.percentile(pixels, [STRETCH_PCT_LOW, STRETCH_PCT_HIGH])
        if hi - lo < 1.0:
            log.warning(
                "%s stretch: band %d degenerate percentile range (lo=%.2f hi=%.2f); widening to lo+1",
                label,
                b,
                float(lo),
                float(hi),
            )
            hi = lo + 1.0
        lows.append(float(lo))
        highs.append(float(hi))
    log.info(
        "%s stretch bounds: lows=%s highs=%s",
        label,
        [round(x, 1) for x in lows],
        [round(x, 1) for x in highs],
    )
    if all(h < STRETCH_HIGH_MIN_SANITY for h in highs):
        raise RuntimeError(
            f"{label} stretch bounds absurdly tight (all highs < {STRETCH_HIGH_MIN_SANITY}: "
            f"{[round(x, 1) for x in highs]}). Refusing to apply; upstream calibration or "
            f"input imagery is pathological."
        )
    return lows, highs


def _apply_stretch_bounds(path: Path, lows: list[float], highs: list[float]) -> None:
    """Rescale each band of `path` in place so [lo, hi] maps to [0, 255]."""
    tmp = path.with_suffix(path.suffix + ".stretch.tmp")
    with rasterio.open(path) as src:
        profile = _gtiff_profile(src.height, src.width, src.crs, src.transform)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band in range(1, 4):
                arr = src.read(band).astype(np.float32)
                nodata = arr == 0
                scaled = (arr - lows[band - 1]) * 255.0 / (highs[band - 1] - lows[band - 1])
                out = np.clip(scaled, 0, 255).astype(np.uint8)
                out[nodata] = 0
                dst.write(out, band)
                del arr, scaled, out, nodata
    tmp.replace(path)


def _convert_to_cog(path: Path) -> None:
    """Rewrite `path` in place as a Cloud Optimized GeoTIFF with baked overviews (GDAL COG driver)."""
    tmp = path.with_suffix(path.suffix + ".cog.tmp")
    creation_options = {
        "compress": "DEFLATE",
        "blocksize": 512,
        "overview_resampling": "average",
        "num_threads": "ALL_CPUS",
        "bigtiff": "YES",
    }
    rio_shutil_copy(str(path), str(tmp), driver="COG", **creation_options)
    tmp.replace(path)
    with rasterio.open(path) as src:
        n_ov = len(src.overviews(1))
        log.info("cog: %s -> %d overview levels, %.1f GB", path.name, n_ov, path.stat().st_size / 1e9)


def _apply_shared_stretch_from_post(pre_aligned: Path, post_aligned: Path) -> None:
    """One 2-98 stretch, computed on post, applied to both; keeps pre tonally locked to post."""
    bounds = _compute_stretch_bounds(post_aligned, "post (shared)")
    if bounds is None:
        return
    _apply_stretch_bounds(post_aligned, *bounds)
    _apply_stretch_bounds(pre_aligned, *bounds)


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
    Image.fromarray(out.transpose(1, 2, 0)).save(out_png)
