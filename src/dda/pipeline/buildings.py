"""Building footprints on `pre_aligned` via fAIr's dinov3-hot-buildings torch path; ONNX is broken on CUDA."""

import gc
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import rasterio
from dinov3_hot.config import load_config
from dinov3_hot.infer import (
    instance_separate,
    load_model,
    sliding_window_predict,
    vectorize,
)
from dinov3_hot.serve import HOT_MEAN, HOT_STD, add_scores
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import Point, box, shape
from shapely.prepared import prep

from dda.hf_utils import resolve_sha_pinned_ckpt
from dda.pipeline.hub_utils import enable_offline_torch_hub_fallback

log = logging.getLogger(__name__)

BUILDING_CKPT_REPO = "kshitijrajsharma/dinov3-hot-buildings"
BUILDING_CKPT_FILE = "dinov3l_upernet_hot.ckpt"
BUILDING_CKPT_SHA256 = "6555c68125ff03298c68e643bc7b8065e81cd7e781ad0d9970281e3c13afa3ce"

# Tuned inference hyperparameters, from the fAIr buildings STAC defaults.
_INFER = {
    "threshold": 0.4371,
    "window": 256,
    "stride": 192,
    "seed_min_distance": 6,
    "large_blob_area_px": 1500,
    "h_maxima_depth": 0.2,
    "min_area_m2": 2.6465,
    "simplify_m": 0.9626,
    "regularize_area_threshold": 0.4949,
    "regularize_overlap_tol_m2": 3.9251,
}


@dataclass
class BuildingsConfig:
    core_px: int = 8192
    halo_px: int = 384
    device: str = "cuda"


def run_fair_buildings(  # noqa: PLR0915  # single-purpose macroblock loop, splitting hurts readability
    pre_aligned: Path,
    aoi: Path,
    out_geojson: Path,
    cfg: BuildingsConfig | None = None,
    ckpt: Path | None = None,
) -> Path:
    """Detect buildings on `pre_aligned` clipped to `aoi`. Writes `out_geojson`; returns that path.

    `ckpt`: optional path to a fine-tuned checkpoint. Defaults to the pretrained HF ckpt.
    """
    cfg = cfg or BuildingsConfig()
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    enable_offline_torch_hub_fallback()
    ckpt = Path(ckpt) if ckpt is not None else _resolve_building_ckpt()

    model_cfg = load_config(None)
    model = load_model(str(ckpt), model_cfg, device=cfg.device)
    log.info("buildings: model loaded from %s", ckpt)

    with rasterio.open(pre_aligned) as s:
        height, width = s.height, s.width
        transform = s.transform
        raster_crs = s.crs
    blocks = [(cx, cy) for cy in range(0, height, cfg.core_px) for cx in range(0, width, cfg.core_px)]
    log.info(
        "buildings: raster %dx%d, %d macroblocks (core=%d, halo=%d)",
        width,
        height,
        len(blocks),
        cfg.core_px,
        cfg.halo_px,
    )

    aoi_geom = _load_aoi_geom(aoi)
    aoi_prep = prep(aoi_geom)

    fd, tmp_name = tempfile.mkstemp(prefix="_block_scratch_", suffix=".tif", dir=out_geojson.parent)
    os.close(fd)
    block_tmp = Path(tmp_name)
    t0 = time.time()
    all_feats: list[dict] = []
    skipped = 0
    try:
        for i, (cx, cy) in enumerate(blocks, start=1):
            cxe = min(cx + cfg.core_px, width)
            cye = min(cy + cfg.core_px, height)
            # Core bounds in 4326: shared frame for the AOI skip and the seam-edge centroid dedup.
            cw_r, cn_r = transform * (cx, cy)
            ce_r, cs_r = transform * (cxe, cye)
            cw, cs, ce, cn = transform_bounds(
                raster_crs,
                "EPSG:4326",
                min(cw_r, ce_r),
                min(cs_r, cn_r),
                max(cw_r, ce_r),
                max(cs_r, cn_r),
            )
            if not aoi_prep.intersects(box(cw, cs, ce, cn)):
                skipped += 1
                if i % 5 == 0 or i == len(blocks):
                    log.info(
                        "buildings: [%d/%d] outside AOI, skipped (%d skipped, %d kept)",
                        i,
                        len(blocks),
                        skipped,
                        len(all_feats),
                    )
                continue

            fx = max(0, cx - cfg.halo_px)
            fy = max(0, cy - cfg.halo_px)
            fxe = min(width, cxe + cfg.halo_px)
            fye = min(height, cye + cfg.halo_px)
            with rasterio.open(pre_aligned) as s:
                win = Window(fx, fy, fxe - fx, fye - fy)  # ty: ignore[too-many-positional-arguments]
                arr = s.read([1, 2, 3], window=win)
                prof = {
                    "driver": "GTiff",
                    "height": arr.shape[1],
                    "width": arr.shape[2],
                    "count": 3,
                    "dtype": "uint8",
                    "crs": s.crs,
                    "transform": s.window_transform(win),
                    "tiled": True,
                    "blockxsize": 512,
                    "blockysize": 512,
                }
                with rasterio.open(block_tmp, "w", **prof) as d:
                    d.write(arr)
            del arr

            mp, _, dist, tr, crs = sliding_window_predict(
                model,
                str(block_tmp),
                window=int(_INFER["window"]),
                stride=int(_INFER["stride"]),
                mean=HOT_MEAN,
                std=HOT_STD,
                device=cfg.device,
            )
            lab = instance_separate(
                mp,
                dist,
                mask_threshold=_INFER["threshold"],
                seed_min_distance=int(_INFER["seed_min_distance"]),
                large_blob_area_px=int(_INFER["large_blob_area_px"]),
                h_maxima_depth=_INFER["h_maxima_depth"],
            )
            gdf = vectorize(
                lab,
                tr,
                crs,
                min_area_m2=_INFER["min_area_m2"],
                simplify_m=_INFER["simplify_m"],
                regularize_area_threshold=_INFER["regularize_area_threshold"],
                regularize_overlap_tol_m2=_INFER["regularize_overlap_tol_m2"],
            )
            if len(gdf):
                scored = add_scores(gdf, lab, mp, tr, crs).to_crs(4326)
                scored = scored.rename(columns={"score": "building_confidence"})
                feats = json.loads(scored[["building_confidence", "geometry"]].to_json())["features"]
            else:
                feats = []
            del mp, dist, lab, gdf
            gc.collect()

            kept = 0
            for f in feats:
                ring = f["geometry"]["coordinates"][0]
                x = sum(p[0] for p in ring) / len(ring)
                y = sum(p[1] for p in ring) / len(ring)
                if cw <= x < ce and cs < y <= cn:
                    all_feats.append(f)
                    kept += 1
            if i % 5 == 0 or i == len(blocks):
                log.info(
                    "buildings: [%d/%d] +%d | total %d (%ds)",
                    i,
                    len(blocks),
                    kept,
                    len(all_feats),
                    int(time.time() - t0),
                )
    finally:
        block_tmp.unlink(missing_ok=True)

    clip = prep(aoi_geom)
    final = [
        f
        for f in all_feats
        if clip.contains(
            Point(
                sum(p[0] for p in f["geometry"]["coordinates"][0]) / len(f["geometry"]["coordinates"][0]),
                sum(p[1] for p in f["geometry"]["coordinates"][0]) / len(f["geometry"]["coordinates"][0]),
            )
        )
    ]
    for i, f in enumerate(final):
        f.setdefault("properties", {})["id"] = i
    import geopandas as gpd

    from dda.pipeline.geowrite import write_dual
    from dda.pipeline.regularise import regularise_footprints

    gdf_final = gpd.GeoDataFrame.from_features(final, crs="EPSG:4326")
    raw_out = out_geojson.with_name(out_geojson.stem + "_raw.geojson")
    gdf_final.to_file(raw_out, driver="GeoJSON")
    log.info("buildings: saved pre-regularise raw -> %s (%d polys)", raw_out, len(gdf_final))
    gdf_final = regularise_footprints(gdf_final, raster_path=str(pre_aligned))
    write_dual(gdf_final, out_geojson)
    log.info(
        "buildings: WROTE %d buildings -> %s(.geojson|.parquet) (%ds)",
        len(final),
        out_geojson.with_suffix(""),
        int(time.time() - t0),
    )
    return out_geojson


def _resolve_building_ckpt() -> Path:
    return resolve_sha_pinned_ckpt(
        BUILDING_CKPT_REPO, BUILDING_CKPT_FILE, BUILDING_CKPT_SHA256, label="buildings ckpt"
    )


def _load_aoi_geom(path: Path):
    """Load an AOI geometry from Feature, FeatureCollection, or bare Geometry GeoJSON."""
    doc = json.loads(Path(path).read_text())
    t = doc.get("type")
    if t == "FeatureCollection":
        return shape(doc["features"][0]["geometry"])
    if t == "Feature":
        return shape(doc["geometry"])
    return shape(doc)
