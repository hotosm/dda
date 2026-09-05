"""Fetch a pre- or post-event raster over an AOI.

Three sources are supported and auto-detected from the URL:
  - Direct COG (http(s) URL ending in .tif, or s3://):    windowed rasterio read
  - Bing Maps aerial (host contains virtualearth.net):    quadkey XYZ, merged
  - Any other TMS URL with {z}/{x}/{y} placeholders:      geomltoolkits.download_tiles
"""

import io
import logging
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import mercantile
import numpy as np
import rasterio
import rasterio.windows
from geomltoolkits import merge_rasters
from PIL import Image
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds as window_from_bounds

log = logging.getLogger(__name__)

BING_TEMPLATE = "https://ecn.t{s}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
# ESRI and some hosts return a placeholder to non-browser UAs; a Mozilla UA bypasses that.
TMS_USER_AGENT = "Mozilla/5.0 (dda-pipeline; contact: krschap@duck.com)"


def fetch_raster(source: str, aoi_geojson: Path, out_path: Path, zoom: int = 19) -> Path:
    """Dispatch to the right backend based on the source URL. Returns `out_path` on success.
    A completed `out_path` skips the fetch; a stray `.part` from a killed prior run is discarded."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        log.info("fetch skipped, output exists: %s (%.1f GB)", out_path, out_path.stat().st_size / 1e9)
        return out_path
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    if part_path.exists():
        log.warning("fetch: discarding stale partial %s", part_path)
        part_path.unlink()
    if _is_cog_url(source):
        _fetch_cog(source, aoi_geojson, part_path)
    elif "virtualearth.net" in source:
        _fetch_bing(aoi_geojson, part_path, zoom=zoom)
    else:
        _fetch_tms(source, aoi_geojson, part_path, zoom=zoom)
    part_path.replace(out_path)
    return out_path


def _is_cog_url(url: str) -> bool:
    lower = url.lower()
    return lower.endswith(".tif") or lower.endswith(".tiff") or url.startswith("s3://")


def _load_bbox(aoi_geojson: Path) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in EPSG:4326 for the AOI."""
    gdf = gpd.read_file(aoi_geojson)
    if gdf.crs is None:
        raise ValueError(f"AOI {aoi_geojson} has no CRS")
    gdf = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gdf.total_bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def _fetch_cog(cog_url: str, aoi_geojson: Path, out_path: Path) -> Path:
    """Window-read the AOI bbox from a COG; intersect with source bounds so the geotransform stays honest."""
    west, south, east, north = _load_bbox(aoi_geojson)
    # http(s) URLs go through GDAL's /vsicurl/ streaming; /vsi..., s3://, and local paths open directly.
    vsi_url = f"/vsicurl/{cog_url}" if cog_url.startswith(("http://", "https://")) else cog_url
    with rasterio.open(vsi_url) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
        requested = window_from_bounds(left, bottom, right, top, src.transform)
        src_win = rasterio.windows.Window(0, 0, src.width, src.height)  # ty: ignore[too-many-positional-arguments]
        try:
            window = requested.intersection(src_win)
        except rasterio.errors.WindowError as exc:
            raise ValueError(f"AOI does not overlap COG {cog_url}: {exc}") from exc
        arr = src.read([1, 2, 3], window=window)
        win_transform = src.window_transform(window)
        profile = {
            "driver": "GTiff",
            "height": arr.shape[1],
            "width": arr.shape[2],
            "count": 3,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": win_transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "bigtiff": "yes",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr)
    log.info("fetched COG window %s -> %s", arr.shape, out_path)
    return out_path


def _fetch_tms(tms_url: str, aoi_geojson: Path, out_path: Path, zoom: int) -> Path:
    """Threaded urllib fetch of AOI-bbox XYZ tiles with a Mozilla UA so ESRI serves real imagery."""
    west, south, east, north = _load_bbox(aoi_geojson)
    tiles = list(mercantile.tiles(west, south, east, north, zooms=[zoom]))
    if not tiles:
        raise ValueError(f"AOI {aoi_geojson} produced 0 tiles at zoom {zoom}")
    log.info("TMS z=%d: %d tiles for AOI", zoom, len(tiles))
    with tempfile.TemporaryDirectory(prefix="dda_tms_") as tmp:
        tiles_dir = Path(tmp)
        _download_xyz_tiles(tms_url, tiles, tiles_dir)
        merge_rasters(str(tiles_dir), str(out_path))
    log.info("fetched TMS tiles zoom=%d -> %s", zoom, out_path)
    return out_path


def _download_xyz_tiles(tms_url: str, tiles: list[mercantile.Tile], tiles_dir: Path) -> None:
    def one(tile: mercantile.Tile) -> None:
        url = tms_url.replace("{z}", str(tile.z)).replace("{x}", str(tile.x)).replace("{y}", str(tile.y))
        req = urllib.request.Request(url, headers={"User-Agent": TMS_USER_AGENT})
        data = _fetch_with_retry(req, tile=tile)
        if data is None:
            return
        image = Image.open(io.BytesIO(data)).convert("RGB")
        web = mercantile.xy_bounds(tile)
        transform = transform_from_bounds(web.left, web.bottom, web.right, web.top, image.width, image.height)
        profile = {
            "driver": "GTiff",
            "height": image.height,
            "width": image.width,
            "count": 3,
            "dtype": "uint8",
            "crs": "EPSG:3857",
            "transform": transform,
            "compress": "jpeg",
            "photometric": "ycbcr",
            "tiled": True,
        }
        arr = np.array(image).transpose(2, 0, 1)
        with rasterio.open(tiles_dir / f"tms_{tile.z}_{tile.x}_{tile.y}.tif", "w", **profile) as dst:
            dst.write(arr)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(one, t) for t in tiles]
        for i, f in enumerate(as_completed(futs), start=1):
            f.result()
            if i % 500 == 0:
                log.info("tms progress %d/%d", i, len(tiles))


def _fetch_with_retry(
    req: urllib.request.Request, tile: mercantile.Tile, max_attempts: int = 5
) -> bytes | None:
    """GET `req` with exponential backoff. Returns response bytes, or `None` for a real 404.
    Retries: ConnectionResetError, urllib URLError, TimeoutError, and HTTP 5xx / 408 / 429.
    Raises the final exception after `max_attempts` if the tile can't be fetched at all."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                return None
            if err.code in (408, 429) or 500 <= err.code < 600:
                last_exc = err
            else:
                raise
        except (urllib.error.URLError, ConnectionResetError, TimeoutError) as err:
            last_exc = err
        if attempt < max_attempts:
            log.warning(
                "tile %d/%d/%d attempt %d failed (%s), retrying in %.1fs",
                tile.z,
                tile.x,
                tile.y,
                attempt,
                type(last_exc).__name__,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(
        f"tile {tile.z}/{tile.x}/{tile.y} failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


def _fetch_bing(aoi_geojson: Path, out_path: Path, zoom: int) -> Path:
    """Bing Maps aerial via quadkey addressing; JPEGs are georeferenced then merged."""
    west, south, east, north = _load_bbox(aoi_geojson)
    tiles = list(mercantile.tiles(west, south, east, north, zooms=[zoom]))
    if not tiles:
        raise ValueError(f"AOI {aoi_geojson} produced 0 tiles at zoom {zoom}")
    log.info("Bing z=%d: %d tiles for AOI", zoom, len(tiles))
    with tempfile.TemporaryDirectory(prefix="dda_bing_") as tmp:
        tiles_dir = Path(tmp)
        for i, tile in enumerate(tiles):
            _download_bing_tile(tile, tiles_dir)
            if (i + 1) % 500 == 0:
                log.info("bing progress %d/%d", i + 1, len(tiles))
        merge_rasters(str(tiles_dir), str(out_path))
    log.info("fetched Bing tiles -> %s", out_path)
    return out_path


def _download_bing_tile(tile: mercantile.Tile, tiles_dir: Path) -> Path:
    """Fetch one Bing quadkey tile as JPEG, write as a georeferenced GeoTIFF in EPSG:3857."""
    quadkey = mercantile.quadkey(tile)
    subdomain = (tile.x + tile.y) % 4
    url = BING_TEMPLATE.format(s=subdomain, q=quadkey)
    request = urllib.request.Request(url, headers={"User-Agent": "dda-pipeline"})
    with urllib.request.urlopen(request, timeout=30) as resp:
        data = resp.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    web = mercantile.xy_bounds(tile)
    tif_path = tiles_dir / f"bing_{tile.z}_{tile.x}_{tile.y}.tif"
    width, height = image.size
    transform = transform_from_bounds(web.left, web.bottom, web.right, web.top, width, height)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 3,
        "dtype": "uint8",
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "jpeg",
        "photometric": "ycbcr",
        "tiled": True,
    }
    arr = np.array(image).transpose(2, 0, 1)
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(arr)
    return tif_path
