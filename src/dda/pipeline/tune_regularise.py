"""Optuna TPE search over RegulariseParams against ground truth; composite objective, no single term wins."""

import logging
import math
import time
from dataclasses import fields
from typing import Any

import geopandas as gpd
import numpy as np
import optuna
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.strtree import STRtree

from dda.pipeline.regularise import RegulariseParams, regularise_footprints

log = logging.getLogger(__name__)

IOU_MATCH_THRESHOLD = 0.5
RATIO_EPS = 1e-6

WEIGHTS = {
    "f1": 0.30,
    "mean_iou": 0.15,
    "poly_count": 0.20,
    "vertex_count": 0.15,
    "orthogonality": 0.10,
    "compactness": 0.10,
}


def tune_regularise(
    raw_gdf: gpd.GeoDataFrame,
    gt_gdf: gpd.GeoDataFrame,
    raster_path: str | None = None,
    n_trials: int = 50,
    seed: int = 42,
) -> tuple[RegulariseParams, dict[str, Any]]:
    """Search 11 quality knobs against gt_gdf; max_iterations and converge_tilted_pct_tol stay at defaults."""
    if not len(raw_gdf):
        raise ValueError("raw_gdf is empty; nothing to tune")
    if not len(gt_gdf):
        raise ValueError("gt_gdf is empty; a ground-truth layer is required")
    if raw_gdf.crs is None or raw_gdf.crs.to_epsg() != 4326:
        raise ValueError(f"raw_gdf must be EPSG:4326, got {raw_gdf.crs}")
    if gt_gdf.crs is None or gt_gdf.crs.to_epsg() != 4326:
        raise ValueError(f"gt_gdf must be EPSG:4326, got {gt_gdf.crs}")

    gt_polys = _extract_polygons(gt_gdf)
    if not gt_polys:
        raise ValueError("gt_gdf contains no valid Polygon geometries")
    gt_compactness_mean = float(np.mean([_isoperimetric_quotient(p) for p in gt_polys]))
    gt_vertex_total = sum(_exterior_vertex_count(p) for p in gt_polys)

    def objective(trial: optuna.Trial) -> float:
        params = RegulariseParams(
            min_area_m2=trial.suggest_float("min_area_m2", 0.5, 15.0),
            simplify_m=trial.suggest_float("simplify_m", 0.5, 4.0),
            simplify_perimeter_pct=trial.suggest_float("simplify_perimeter_pct", 0.01, 0.06),
            reflex_min_notch_m=trial.suggest_float("reflex_min_notch_m", 0.3, 1.0),
            ortho_min_area_ratio=trial.suggest_float("ortho_min_area_ratio", 0.55, 0.90),
            ortho_45_tol_deg=trial.suggest_float("ortho_45_tol_deg", 15.0, 40.0),
            pool_min_fraction=trial.suggest_float("pool_min_fraction", 0.30, 0.70),
            sliver_max_area_m2=trial.suggest_float("sliver_max_area_m2", 3.0, 15.0),
            sliver_max_aspect=trial.suggest_float("sliver_max_aspect", 2.5, 6.0),
            multiblob_open_m=trial.suggest_float("multiblob_open_m", 0.5, 4.0),
            fidelity_min_iou=trial.suggest_float("fidelity_min_iou", 0.5, 0.85),
        )
        pred_gdf = regularise_footprints(raw_gdf, params=params, raster_path=raster_path)
        pred_polys = _extract_polygons(pred_gdf)
        if not pred_polys:
            return 0.0
        return _composite_score(pred_polys, gt_polys, gt_compactness_mean, gt_vertex_total)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    started = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - started

    best_params_dict = dict(study.best_params)
    best_params = _merge_defaults(best_params_dict)
    report = {
        "best_value": float(study.best_value),
        "best_params": best_params_dict,
        "n_trials": len(study.trials),
        "elapsed_s": elapsed,
    }
    return best_params, report


def _composite_score(
    pred: list[Polygon],
    gt: list[Polygon],
    gt_compactness_mean: float,
    gt_vertex_total: int,
) -> float:
    """Weighted mix; every term is in [0, 1] with 1.0 as the ideal."""
    f1, mean_iou = _instance_f1_and_mean_iou(pred, gt, IOU_MATCH_THRESHOLD)
    poly_ratio_score = _ratio_score(len(pred) / max(len(gt), 1))
    pred_vertex_total = sum(_exterior_vertex_count(p) for p in pred)
    vertex_ratio_score = _ratio_score(pred_vertex_total / max(gt_vertex_total, 1))
    ortho_score = 1.0 - _orthogonality_norm(pred)
    compactness_score = 1.0 - _compactness_delta_norm(pred, gt_compactness_mean)
    return (
        WEIGHTS["f1"] * f1
        + WEIGHTS["mean_iou"] * mean_iou
        + WEIGHTS["poly_count"] * poly_ratio_score
        + WEIGHTS["vertex_count"] * vertex_ratio_score
        + WEIGHTS["orthogonality"] * ortho_score
        + WEIGHTS["compactness"] * compactness_score
    )


def _extract_polygons(gdf: gpd.GeoDataFrame) -> list[Polygon]:
    return [
        g
        for g in gdf.geometry
        if g is not None and not g.is_empty and g.geom_type == "Polygon" and g.is_valid
    ]


def _instance_f1_and_mean_iou(
    pred: list[Polygon], gt: list[Polygon], iou_threshold: float
) -> tuple[float, float]:
    """Greedy one-to-one pred/gt matching at iou_threshold; unmatched pred = FP, unmatched gt = FN."""
    if not pred or not gt:
        return 0.0, 0.0
    tree = STRtree(gt)
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for jj in tree.query(p):
            j = int(jj)
            iou = _iou(p, gt[j])
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    for iou, i, j in pairs:
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        matched_ious.append(iou)
    tp = len(matched_ious)
    fp = len(pred) - tp
    fn = len(gt) - tp
    denom = 2 * tp + fp + fn
    f1 = (2.0 * tp / denom) if denom > 0 else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return f1, mean_iou


def _iou(a: Polygon, b: Polygon) -> float:
    try:
        inter = a.intersection(b).area
        if inter == 0.0:
            return 0.0
        union = a.union(b).area
    except GEOSException:
        # GEOS boolean op failed on trial geometry: treat as no match, penalises the trial.
        return 0.0
    if union == 0.0:
        return 0.0
    return inter / union


def _ratio_score(ratio: float) -> float:
    """exp(-|log(ratio)|); peaks at 1.0, symmetric under/over penalty, clipped for stability."""
    r = max(min(ratio, 3.0), RATIO_EPS)
    return math.exp(-abs(math.log(r)))


def _exterior_vertex_count(poly: Polygon) -> int:
    return max(len(poly.exterior.coords) - 1, 0)


def _isoperimetric_quotient(poly: Polygon) -> float:
    """perim^2 / (4 pi area); 1.0 for a circle, ~1.27 for a square, grows with irregularity."""
    area = poly.area
    if area <= 0:
        return 0.0
    return (poly.length**2) / (4.0 * math.pi * area)


def _compactness_delta_norm(pred: list[Polygon], gt_compactness_mean: float) -> float:
    """Relative deviation of pred mean compactness from gt mean, clipped to [0, 1]."""
    if not pred or gt_compactness_mean <= 0:
        return 1.0
    pred_mean = float(np.mean([_isoperimetric_quotient(p) for p in pred]))
    delta = abs(pred_mean - gt_compactness_mean) / gt_compactness_mean
    return min(delta, 1.0)


def _orthogonality_norm(pred: list[Polygon]) -> float:
    """Mean |90 - convex interior angle| over all vertices, divided by 90; in [0, 1]."""
    total_dev = 0.0
    total_count = 0
    for poly in pred:
        for angle in _convex_interior_angles_deg(poly):
            total_dev += abs(90.0 - angle)
            total_count += 1
    if total_count == 0:
        return 1.0
    mean_dev = total_dev / total_count
    return min(mean_dev / 90.0, 1.0)


def _convex_interior_angles_deg(poly: Polygon) -> list[float]:
    """Convex interior angles in degrees, one per exterior vertex (reflex angles skipped)."""
    ring = list(poly.exterior.coords[:-1])
    n = len(ring)
    if n < 3:
        return []
    is_ccw = poly.exterior.is_ccw
    out: list[float] = []
    for i in range(n):
        prev = ring[(i - 1) % n]
        curr = ring[i]
        nxt = ring[(i + 1) % n]
        e1x, e1y = curr[0] - prev[0], curr[1] - prev[1]
        e2x, e2y = nxt[0] - curr[0], nxt[1] - curr[1]
        l1 = math.hypot(e1x, e1y)
        l2 = math.hypot(e2x, e2y)
        if l1 == 0.0 or l2 == 0.0:
            continue
        cross = (e1x * e2y - e1y * e2x) / (l1 * l2)
        dot = (e1x * e2x + e1y * e2y) / (l1 * l2)
        turn = math.degrees(math.atan2(cross, dot))
        if not is_ccw:
            turn = -turn
        interior = 180.0 - turn
        if 0.0 < interior < 180.0:
            out.append(interior)
    return out


def _merge_defaults(overrides: dict[str, Any]) -> RegulariseParams:
    """Fill in the two non-tuned knobs (max_iterations, converge_tilted_pct_tol)."""
    kwargs = {f.name: overrides[f.name] for f in fields(RegulariseParams) if f.name in overrides}
    return RegulariseParams(**kwargs)
