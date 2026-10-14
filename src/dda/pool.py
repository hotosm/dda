"""Assign each footprint a damage class + confidence from a dense (C, H, W) probability map (torch-free)."""

import logging
import math
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.windows
from rasterio.features import geometry_mask

from dda.config import DAMAGE_CLASSES, N_DAMAGE_CLASSES

log = logging.getLogger(__name__)

NO_DATA_CLASS = -1
NO_DATA_LABEL = "no-data"
# Minimum share of imaged pixels for a building or labelled area to be scored or trained on.
MIN_COVERAGE_FRACTION = 0.9
_ORDINAL = np.arange(N_DAMAGE_CLASSES)


def _pool_one(pix: np.ndarray, pool_op: str, percentile: float) -> int:
    """pix is (C, n_pixels) of per-class probabilities; returns the picked class."""
    if pool_op == "mean":
        return int(pix.mean(axis=1).argmax())
    if pool_op == "max":
        return int(pix.argmax(axis=0).max())
    severity = (_ORDINAL[:, None] * pix).sum(axis=0)
    sev = float(np.percentile(severity, percentile))
    return int(np.clip(round(sev), 0, N_DAMAGE_CLASSES - 1))


def assign_damage(
    prob: np.ndarray,
    transform: "rasterio.Affine",
    crs: Any,
    buildings: gpd.GeoDataFrame,
    pool_op: str = "percentile",
    percentile: float = 80.0,
) -> gpd.GeoDataFrame:
    """prob is (C, H, W) softmax; footprints get damage_class, damage, damage_confidence."""
    height, width = prob.shape[1:]
    inv = ~transform
    out = buildings.to_crs(crs).copy()
    classes, labels, confs = [], [], []

    for geom in out.geometry:
        minx, miny, maxx, maxy = geom.bounds
        c_tl, r_tl = inv * (minx, maxy)
        c_br, r_br = inv * (maxx, miny)
        c0, c1 = max(0, int(c_tl)), min(width, int(c_br) + 1)
        r0, r1 = max(0, int(r_tl)), min(height, int(r_br) + 1)
        if c1 <= c0 or r1 <= r0:
            classes.append(NO_DATA_CLASS)
            labels.append("no-data")
            confs.append(float("nan"))
            continue
        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)  # ty: ignore[too-many-positional-arguments]
        win_t = rasterio.windows.transform(window, transform)
        inside = geometry_mask([geom], out_shape=(r1 - r0, c1 - c0), transform=win_t, invert=True)
        if not inside.any():
            classes.append(NO_DATA_CLASS)
            labels.append("no-data")
            confs.append(float("nan"))
            continue
        pix = prob[:, r0:r1, c0:c1][:, inside]
        cls = _pool_one(pix, pool_op, percentile)
        classes.append(cls)
        labels.append(DAMAGE_CLASSES[cls])
        confs.append(float(pix[cls].mean()))

    out["damage_class"] = classes
    out["damage"] = labels
    out["damage_confidence"] = confs
    log.info("Assigned damage to %d buildings", len(out))
    return out


def parse_confidence_threshold(value: object) -> float:
    """Raise ValueError unless value is a number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confidence_threshold must be a number, got {value!r}")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"confidence_threshold must be within [0, 1], got {threshold}")
    return threshold


def filter_by_confidence(buildings: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    """No-data buildings (NaN confidence) are kept."""
    confidence = buildings["damage_confidence"]
    return buildings[confidence.isna() | (confidence >= threshold)]


def _footprint_mask(geom: Any, transform: "rasterio.Affine") -> tuple[int, int, np.ndarray]:
    """Top-left row and column of the footprint's bounding window, unclipped, and its inside mask."""
    minx, miny, maxx, maxy = geom.bounds
    c_tl, r_tl = ~transform * (minx, maxy)
    c_br, r_br = ~transform * (maxx, miny)
    r0, c0 = math.floor(r_tl), math.floor(c_tl)
    r1, c1 = math.floor(r_br) + 1, math.floor(c_br) + 1
    window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)  # ty: ignore[too-many-positional-arguments]
    window_transform = rasterio.windows.transform(window, transform)
    inside = geometry_mask([geom], out_shape=(r1 - r0, c1 - c0), transform=window_transform, invert=True)
    return r0, c0, inside


def coverage_fraction(buildings: gpd.GeoDataFrame, valid: np.ndarray, transform) -> np.ndarray:
    """Per-building fraction of `valid` pixels, with pixels off the grid counted as not valid; each footprint
    is measured alone so overlaps do not hide one another. Buildings must be in the grid's CRS."""
    coverage = np.zeros(len(buildings), dtype=np.float32)
    height, width = valid.shape
    for index, geom in enumerate(buildings.geometry):
        r0, c0, inside = _footprint_mask(geom, transform)
        rr0, cc0 = max(r0, 0), max(c0, 0)
        rr1, cc1 = min(r0 + inside.shape[0], height), min(c0 + inside.shape[1], width)
        if not inside.any() or rr1 <= rr0 or cc1 <= cc0:
            continue
        imaged = np.zeros(inside.shape, dtype=bool)
        imaged[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] = valid[rr0:rr1, cc0:cc1]
        coverage[index] = imaged[inside].mean()
    return coverage


def mark_uncovered_as_no_data(scored: gpd.GeoDataFrame, imaged: np.ndarray, transform) -> gpd.GeoDataFrame:
    """`scored` with buildings under `MIN_COVERAGE_FRACTION` of `imaged` pixels set to the no-data class."""
    out = scored.copy()
    uncovered = coverage_fraction(out, imaged, transform) < MIN_COVERAGE_FRACTION
    out.loc[uncovered, "damage_class"] = NO_DATA_CLASS
    out.loc[uncovered, "damage"] = NO_DATA_LABEL
    out.loc[uncovered, "damage_confidence"] = np.nan
    return out
