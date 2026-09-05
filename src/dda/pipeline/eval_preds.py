"""Per-class precision/recall/F1 + confusion matrix against a labelled GeoJSON matched by centroid."""

import json
import logging
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
MATCH_RADIUS_M = 5.0


def evaluate(predictions: Path, labels: Path, out_json: Path | None = None) -> dict:
    """Compute per-class F1 + confusion matrix + accuracy. Returns a JSON-serialisable dict."""
    preds = gpd.read_file(predictions).to_crs("EPSG:4326")
    gt = gpd.read_file(labels).to_crs("EPSG:4326")
    if "damage" not in preds.columns:
        raise ValueError(f"{predictions} has no `damage` column")
    if "damage" not in gt.columns:
        raise ValueError(f"{labels} has no `damage` column")

    # Metric CRS so the match radius is metres.
    preds_m = preds.to_crs("EPSG:3857")
    gt_m = gt.to_crs("EPSG:3857")
    pc = np.array([(g.centroid.x, g.centroid.y) for g in preds_m.geometry])
    gc = np.array([(g.centroid.x, g.centroid.y) for g in gt_m.geometry])
    if len(pc) == 0 or len(gc) == 0:
        raise ValueError(f"empty inputs: {len(pc)} preds, {len(gc)} labels")

    tree = cKDTree(pc)
    dist, idx = tree.query(gc, distance_upper_bound=MATCH_RADIUS_M)
    matched = dist != np.inf
    unmatched = int((~matched).sum())
    log.info(
        "matched %d / %d labels (%.1f%%) within %.0f m",
        matched.sum(),
        len(gc),
        100 * matched.mean(),
        MATCH_RADIUS_M,
    )

    y_true = gt["damage"].values[matched]
    y_pred = preds["damage"].values[idx[matched]]

    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    cls_ix = {c: i for i, c in enumerate(CLASSES)}
    for t, p in zip(y_true, y_pred, strict=True):
        if t in cls_ix and p in cls_ix:
            cm[cls_ix[t], cls_ix[p]] += 1

    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1": round(f1, 3),
            "support": int(cm[i, :].sum()),
        }

    total = int(cm.sum())
    accuracy = float(np.trace(cm)) / total if total else 0.0

    result = {
        "n_predictions": len(preds),
        "n_labels": len(gt),
        "n_matched": int(matched.sum()),
        "n_unmatched": unmatched,
        "match_radius_m": MATCH_RADIUS_M,
        "accuracy": round(accuracy, 3),
        "macro_f1": round(np.mean([per_class[c]["f1"] for c in CLASSES]), 3),
        "per_class": per_class,
        "confusion_matrix": {"classes": CLASSES, "matrix": cm.tolist()},
        "predicted_class_distribution": dict(Counter(preds["damage"].astype(str))),
        "label_class_distribution": dict(Counter(gt["damage"].astype(str))),
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
        log.info("wrote %s", out_json)
    return result
