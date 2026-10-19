"""Buildings with under 90% pre or post coverage get class -1 so tile-mosaic gaps do not
score as confident 'destroyed'."""

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from omegaconf import DictConfig
from rasterio.windows import Window

from dda.infer import load_model, sliding_window_prob
from dda.pool import MIN_COVERAGE_FRACTION, NO_DATA_CLASS, NO_DATA_LABEL, assign_damage, coverage_fraction
from dda.tiling import radiometric_normalize

log = logging.getLogger(__name__)


@dataclass
class DamageConfig:
    core_px: int = 4096
    halo_px: int = 512
    pre_coverage_threshold: float = MIN_COVERAGE_FRACTION
    post_coverage_threshold: float = MIN_COVERAGE_FRACTION


def run_damage_blocked(
    cfg: DictConfig,
    ckpt_path: str | Path,
    post_raster: Path,
    pre_aligned: Path,
    buildings_geojson: Path,
    out_geojson: Path,
    damage_cfg: DamageConfig | None = None,
    device: str | None = None,
) -> Path:
    """Run the model in core + halo blocks and pool damage per building. Returns `out_geojson`."""
    damage_cfg = damage_cfg or DamageConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, cfg, device=device)

    buildings = gpd.read_file(buildings_geojson)
    with rasterio.open(post_raster) as post_src:
        post_crs = post_src.crs
        post_transform = post_src.transform
        post_height, post_width = post_src.height, post_src.width

    buildings_in_post = buildings.to_crs(post_crs)
    buildings_in_post["centroid_px"] = buildings_in_post.geometry.centroid.map(
        lambda p: (~post_transform) * (p.x, p.y)
    )

    core = damage_cfg.core_px
    halo = damage_cfg.halo_px
    blocks = [(cx, cy) for cy in range(0, post_height, core) for cx in range(0, post_width, core)]
    log.info(
        "damage %dx%d px, %d blocks (core=%d, halo=%d), %d buildings",
        post_width,
        post_height,
        len(blocks),
        core,
        halo,
        len(buildings_in_post),
    )

    accum: list[gpd.GeoDataFrame] = []
    for i, (cx, cy) in enumerate(blocks, start=1):
        block_gdf = _process_block(
            cfg=cfg,
            model=model,
            post_raster=post_raster,
            pre_aligned=pre_aligned,
            buildings=buildings_in_post,
            block_x=cx,
            block_y=cy,
            core=core,
            halo=halo,
            post_width=post_width,
            post_height=post_height,
            damage_cfg=damage_cfg,
            device=device,
        )
        log.info("block %d/%d -> %d buildings", i, len(blocks), len(block_gdf))
        if len(block_gdf) > 0:
            accum.append(block_gdf)

    out = pd.concat(accum, ignore_index=True) if accum else buildings_in_post.iloc[0:0].copy()
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=post_crs).to_crs(buildings.crs)
    out = out.drop(columns=[c for c in ["centroid_px"] if c in out.columns])
    out = _apply_output_schema(out, cfg)  # ty: ignore[invalid-argument-type]
    from dda.pipeline.geowrite import write_dual

    write_dual(out, out_geojson)
    log.info("wrote %d buildings with damage -> %s(.geojson|.parquet)", len(out), out_geojson.with_suffix(""))
    return out_geojson


def _apply_output_schema(gdf: gpd.GeoDataFrame, cfg: DictConfig) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    label_map = {int(k): str(v) for k, v in cfg.damage_label_map.items()}
    gdf["damage"] = gdf["damage_class"].map(label_map)
    gdf["damage_model"] = cfg.damage_provenance_damage_model
    gdf["imagery_pre"] = cfg.damage_provenance_imagery_pre
    gdf["imagery_post"] = cfg.damage_provenance_imagery_post
    schema = [c for c in cfg.damage_output_schema if c in gdf.columns]
    if "geometry" not in schema:
        schema.append("geometry")
    return gdf[schema]


def _process_block(
    *,
    cfg: DictConfig,
    model,
    post_raster: Path,
    pre_aligned: Path,
    buildings: gpd.GeoDataFrame,
    block_x: int,
    block_y: int,
    core: int,
    halo: int,
    post_width: int,
    post_height: int,
    damage_cfg: DamageConfig,
    device: str,
) -> gpd.GeoDataFrame:
    """One block: assign each contained-in-core building a class + confidence."""
    core_x0, core_y0 = block_x, block_y
    core_x1, core_y1 = min(block_x + core, post_width), min(block_y + core, post_height)
    in_core = buildings["centroid_px"].map(
        lambda cp: core_x0 <= cp[0] < core_x1 and core_y0 <= cp[1] < core_y1
    )
    block_buildings = buildings.loc[in_core].copy()
    if len(block_buildings) == 0:
        return _empty_from_template(buildings)

    read_x0 = max(0, core_x0 - halo)
    read_y0 = max(0, core_y0 - halo)
    read_x1 = min(post_width, core_x1 + halo)
    read_y1 = min(post_height, core_y1 + halo)
    window = Window(read_x0, read_y0, read_x1 - read_x0, read_y1 - read_y0)  # ty: ignore[too-many-positional-arguments]

    with rasterio.open(post_raster) as post_src:
        post_arr = post_src.read([1, 2, 3], window=window).transpose(1, 2, 0).astype(np.uint8)
        block_transform = post_src.window_transform(window)
    with rasterio.open(pre_aligned) as pre_src:
        pre_arr = pre_src.read([1, 2, 3], window=window).transpose(1, 2, 0).astype(np.uint8)

    if cfg.radiometric_normalize:
        if post_arr.any():
            post_arr = radiometric_normalize(post_arr)
        if pre_arr.any():
            pre_arr = radiometric_normalize(pre_arr)

    pre_valid = pre_arr.any(axis=2)
    post_valid = post_arr.any(axis=2)
    pre_frac = coverage_fraction(block_buildings, pre_valid, block_transform)
    post_frac = coverage_fraction(block_buildings, post_valid, block_transform)
    pre_ok = pre_frac >= damage_cfg.pre_coverage_threshold
    post_ok = post_frac >= damage_cfg.post_coverage_threshold
    valid_mask = pre_ok & post_ok

    covered_buildings = block_buildings.loc[valid_mask].copy()
    no_pre_buildings = _label_no_data(block_buildings.loc[~pre_ok].copy(), NO_DATA_LABEL)
    no_post_buildings = _label_no_data(block_buildings.loc[pre_ok & ~post_ok].copy(), NO_DATA_LABEL)

    scored = _empty_from_template(block_buildings)
    if len(covered_buildings) > 0:
        prob = sliding_window_prob(
            model,
            post_arr,
            pre_arr,
            cfg.tile_window,
            cfg.tile_stride,
            cfg.temperature,
            device,
        )
        scored = assign_damage(
            prob,
            block_transform,
            covered_buildings.crs,
            covered_buildings,
            pool_op=cfg.pool_op,
            percentile=cfg.pool_percentile,
        )

    merged = pd.concat([scored, no_pre_buildings, no_post_buildings], ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=scored.crs)


def _label_no_data(buildings: gpd.GeoDataFrame, label: str) -> gpd.GeoDataFrame:
    """Attach the no-data class + label; damage_confidence is NaN (model never ran)."""
    out = buildings.copy()
    out["damage_class"] = NO_DATA_CLASS
    out["damage"] = label
    out["damage_confidence"] = np.nan
    return out


def _empty_from_template(template: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preserves int dtype of osm_id; concat with an all-NaN placeholder would promote it to float64."""
    empty = template.iloc[0:0].copy()
    for col, dtype in (("damage_class", "int64"), ("damage", object), ("damage_confidence", "float64")):
        if col not in empty.columns:
            empty[col] = pd.Series(dtype=dtype)
    return empty
