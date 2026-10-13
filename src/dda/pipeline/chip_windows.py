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

from dda.infer import radiometric_normalize
from dda.pipeline.fetch import fetch_raster
from dda.pipeline.fewshot_damage import normalize_damage_value
from dda.pipeline.osm_pull import pull_osm_buildings

CHIP_NAME = re.compile(r"OAM-(\d+)-(\d+)-(\d+)\.tif")
OSM_BUILDING_FAMILIES = [{"key": "building", "status": "standing"}]


def parse_chip_name(name: str) -> tuple[int, int, int]:
    match = CHIP_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"chip name {name!r} does not match OAM-{{x}}-{{y}}-{{z}}.tif")
    x, y, z = (int(value) for value in match.groups())
    return x, y, z


def windows_from_chips(
    post_dir: Path, pre_dir: Path, labels: gpd.GeoDataFrame, chip_names: Iterable[str]
) -> list[dict[str, Any]]:
    """One window per 2x2 block of tiles that holds a labelled building, in the format
    `fit_damage_windows` takes. `labels` has a `damage` column of xBD names or codes 1..4; post and
    pre are stretched per window, as `read_aligned_pair` stretches the inference area."""
    blocks: dict[tuple[int, int, int], list[tuple[str, int, int]]] = defaultdict(list)
    for name in chip_names:
        x, y, z = parse_chip_name(name)
        blocks[(x // 2, y // 2, z)].append((name, x % 2, y % 2))
    codes = [normalize_damage_value(value) for value in labels["damage"]]

    windows = []
    for members in (blocks[key] for key in sorted(blocks)):
        post, pre, transform, crs = _read_block(Path(post_dir), Path(pre_dir), members)
        shapes = zip(labels.to_crs(crs).geometry, codes, strict=True)
        damage = rasterize(shapes, out_shape=post.shape[1:], transform=transform, fill=0, dtype="uint8")
        if not damage.any():
            continue
        windows.append(
            {
                "post": radiometric_normalize(post.transpose(1, 2, 0)).transpose(2, 0, 1),
                "pre": radiometric_normalize(pre.transpose(1, 2, 0)).transpose(2, 0, 1),
                "damage": damage,
            }
        )
    return windows


def _read_block(
    post_dir: Path, pre_dir: Path, members: list[tuple[str, int, int]]
) -> tuple[np.ndarray, np.ndarray, "rasterio.Affine", Any]:
    """Post and pre RGB of a 2x2 tile block, zero where a tile is absent, on the post chips' grid."""
    first, first_col, first_row = members[0]
    with rasterio.open(post_dir / first) as src:
        size = src.width
        transform = src.transform * rasterio.Affine.translation(-first_col * size, -first_row * size)
        crs = src.crs
    post = np.zeros((3, 2 * size, 2 * size), dtype=np.uint8)
    pre = np.zeros_like(post)
    for name, col, row in members:
        cells = np.s_[:, row * size : (row + 1) * size, col * size : (col + 1) * size]
        with rasterio.open(post_dir / name) as post_src, rasterio.open(pre_dir / name) as pre_src:
            post[cells] = post_src.read([1, 2, 3])
            pre[cells] = pre_src.read([1, 2, 3], out_shape=(3, size, size))
    return post, pre, transform, crs


def area_inputs(
    post_chips_dir: Path, pre_image_uri: str, buildings_uri: str | None, work_dir: Path
) -> tuple[Path, Path, gpd.GeoDataFrame]:
    """Post mosaic of the chips, the pre image fetched over its bounds at the chips' zoom, and the
    footprints from `buildings_uri`, or from OpenStreetMap when it is None."""
    chips = sorted(Path(post_chips_dir).rglob("OAM-*.tif"))
    if not chips:
        raise FileNotFoundError(f"no OAM-*.tif chips under {post_chips_dir}")
    zoom = parse_chip_name(chips[0].name)[2]
    work_dir.mkdir(parents=True, exist_ok=True)
    post_raster = Path(merge_rasters([str(chip) for chip in chips], str(work_dir / "post.tif")))
    with rasterio.open(post_raster) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    aoi = work_dir / "aoi.geojson"
    aoi.write_text(json.dumps({"type": "FeatureCollection", "features": [_feature(box(*bounds))]}))
    pre_raster = fetch_raster(pre_image_uri, aoi, work_dir / "pre.tif", zoom=zoom)
    if buildings_uri is None:
        buildings = pull_osm_buildings(aoi, work_dir / "osm", OSM_BUILDING_FAMILIES)
    else:
        buildings = gpd.read_file(buildings_uri, bbox=bounds)
    return post_raster, pre_raster, buildings


def _feature(geometry) -> dict[str, Any]:
    return {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}
