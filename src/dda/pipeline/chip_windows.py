"""Training windows and inference inputs built from fAIr dataset chips named `OAM-{x}-{y}-{z}.tif`."""

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from geomltoolkits import merge_rasters
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

from dda.config import normalize_damage_value
from dda.pipeline.osm_pull import pull_osm_buildings
from dda.pool import MIN_COVERAGE_FRACTION
from dda.tiling import radiometric_normalize

CHIP_NAME = re.compile(r"OAM-(\d+)-(\d+)-(\d+)\.tif")


def parse_chip_name(name: str) -> tuple[int, int, int]:
    match = CHIP_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"chip name {name!r} does not match OAM-{{x}}-{{y}}-{{z}}.tif")
    x, y, z = (int(value) for value in match.groups())
    return x, y, z


def windows_from_chips(
    post_dir: Path, pre_dir: Path, labels: gpd.GeoDataFrame, chip_names: Iterable[str], cfg
) -> list[dict[str, Any]]:
    """One `fit_damage_windows` window per 2x2 tile block whose labels are at least `MIN_COVERAGE_FRACTION`
    imaged in post and pre; off-imagery labels become background; stretched if `cfg.radiometric_normalize`."""
    post_dir, pre_dir = Path(post_dir), Path(pre_dir)
    blocks: dict[tuple[int, int, int], list[tuple[str, int, int]]] = defaultdict(list)
    for name in chip_names:
        x, y, z = parse_chip_name(name)
        blocks[(x // 2, y // 2, z)].append((name, x % 2, y % 2))
    codes = [normalize_damage_value(value) for value in labels["damage"]]

    windows = []
    for key in sorted(blocks):
        post, pre, transform, crs = _read_block(post_dir, pre_dir, blocks[key], cfg.tile_window // 2)
        shapes = zip(labels.to_crs(crs).geometry, codes, strict=True)
        damage = rasterize(shapes, out_shape=post.shape[1:], transform=transform, fill=0, dtype="uint8")
        labelled = damage > 0
        imaged = post.any(axis=0) & pre.any(axis=0)
        if not labelled.any() or imaged[labelled].mean() < MIN_COVERAGE_FRACTION:
            continue
        damage[~imaged] = 0
        if cfg.radiometric_normalize:
            post = radiometric_normalize(post.transpose(1, 2, 0)).transpose(2, 0, 1)
            pre = radiometric_normalize(pre.transpose(1, 2, 0)).transpose(2, 0, 1)
        windows.append({"post": post, "pre": pre, "damage": damage})
    return windows


def _read_block(
    post_dir: Path, pre_dir: Path, members: list[tuple[str, int, int]], chip_size: int
) -> tuple[np.ndarray, np.ndarray, "rasterio.Affine", Any]:
    """Post and pre RGB of a 2x2 tile block, zero where a tile is absent, on the post chips' grid."""
    first, first_col, first_row = members[0]
    with rasterio.open(post_dir / first) as src:
        offset = rasterio.Affine.translation(-first_col * chip_size, -first_row * chip_size)
        transform = src.transform * offset
        crs = src.crs
    post = np.zeros((3, 2 * chip_size, 2 * chip_size), dtype=np.uint8)
    pre = np.zeros_like(post)
    for name, col, row in members:
        cells = np.s_[:, row * chip_size : (row + 1) * chip_size, col * chip_size : (col + 1) * chip_size]
        with rasterio.open(post_dir / name) as post_src, rasterio.open(pre_dir / name) as pre_src:
            if (post_src.width, post_src.height) != (chip_size, chip_size):
                raise ValueError(
                    f"chip {name} is {post_src.width}x{post_src.height} px; the model window takes "
                    f"2x2 blocks of {chip_size} px chips"
                )
            same_area = np.allclose(pre_src.bounds, post_src.bounds, rtol=0, atol=post_src.res[0] / 2)
            if pre_src.crs != post_src.crs or not same_area:
                raise ValueError(f"pre chip {name} does not cover the same area as its post chip")
            post[cells] = post_src.read([1, 2, 3])
            pre[cells] = pre_src.read([1, 2, 3], out_shape=(3, chip_size, chip_size))
    return post, pre, transform, crs


def area_inputs(
    post_chips_dir: Path, pre_chips_dir: Path, buildings_uri: str | None, work_dir: Path, cfg
) -> tuple[Path, Path, gpd.GeoDataFrame]:
    """Post and pre mosaics of the chips, and the footprints over the post mosaic from `buildings_uri`,
    or from OpenStreetMap with `cfg.osm_tag_families` when it is None."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    post_raster = _mosaic_chips(post_chips_dir, work_dir / "post.tif")
    pre_raster = _mosaic_chips(pre_chips_dir, work_dir / "pre.tif")
    with rasterio.open(post_raster) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    aoi = work_dir / "aoi.geojson"
    aoi.write_text(json.dumps({"type": "FeatureCollection", "features": [_feature(box(*bounds))]}))
    if buildings_uri is None:
        families = [dict(family) for family in cfg.osm_tag_families]
        buildings = pull_osm_buildings(aoi, work_dir / "osm", families)
    else:
        buildings = gpd.read_file(buildings_uri, mask=gpd.GeoSeries([box(*bounds)], crs="EPSG:4326"))
    return post_raster, pre_raster, buildings


def _mosaic_chips(chips_dir: Path, out_path: Path) -> Path:
    chips = sorted(Path(chips_dir).glob("OAM-*.tif"))
    if not chips:
        raise FileNotFoundError(f"no OAM-*.tif chips in {chips_dir}")
    return Path(merge_rasters([str(chip) for chip in chips], str(out_path)))


def _feature(geometry) -> dict[str, Any]:
    return {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
