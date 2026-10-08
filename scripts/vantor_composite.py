"""Cloud-native Vantor composite. Streams STAC scenes, dask-parallel, no local cache.

Priority-fallback along the time axis: scenes are sorted by STAC `eo:cloud_cover`
ascending and merged with `odc.stac.load(fuse_func=first_valid)` so the first
non-nodata pixel wins. Output is a single-pass Cloud-Optimised GeoTIFF written
by GDAL's COG driver.
"""

import argparse
import json
import logging
import os
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import odc.stac
import pystac
import rioxarray  # noqa: F401  # registers .rio accessor on xarray
from dask.distributed import Client, LocalCluster

log = logging.getLogger(__name__)

STAC_BASE = "https://vantor-opendata.s3.amazonaws.com/events"

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "5000000",
    "GDAL_CACHEMAX": "200",
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def load_items(event: str, phase: str) -> list[pystac.Item]:
    coll = json.loads(urllib.request.urlopen(f"{STAC_BASE}/{event}/collection.json", timeout=30).read())
    items: list[pystac.Item] = []
    for link in coll.get("links", []):
        if link.get("rel") != "item":
            continue
        href = link["href"]
        if not href.startswith("http"):
            href = f"{STAC_BASE}/{event}/{Path(href).name}"
        raw = json.loads(urllib.request.urlopen(href, timeout=30).read())
        it = pystac.Item.from_dict(raw, href=href)
        if it.properties.get("phase") == phase:
            items.append(it)
    return sorted(items, key=lambda i: i.properties.get("eo:cloud_cover", 100))


def composite(
    items: list[pystac.Item],
    bounds: tuple[float, float, float, float],
    resolution: float,
    out: Path,
    chunksize: int,
    asset: str,
) -> None:
    for k, v in GDAL_ENV.items():
        os.environ[k] = str(v)

    # Vantor items don't declare `raster:bands`, so odc-stac would only see band 1.
    # Inject the 3-band descriptor per item so meta.bands registers all 3, then
    # request them by `<asset>.<bidx>`.
    for it in items:
        it.assets[asset].extra_fields["raster:bands"] = [
            {"nodata": 0, "data_type": "uint8"} for _ in range(3)
        ]

    ds = odc.stac.load(
        items,
        bands=[f"{asset}.1", f"{asset}.2", f"{asset}.3"],
        crs="EPSG:4326",
        resolution=resolution,
        bbox=bounds,
        chunks={"x": chunksize, "y": chunksize},
        resampling="bilinear",
        dtype="uint8",
        nodata=0,
        # Single bucket, so the default fuser (first non-nodata wins) picks across every
        # scene. Since items are sorted cloud-asc, this yields a cloud-priority mosaic.
        groupby=lambda _item, _parsed, _idx: "all",
    )
    log.info("loaded: %s", {k: v.shape for k, v in ds.data_vars.items()})

    da = ds.to_array(dim="band")
    if "time" in da.dims:
        da = da.isel(time=0)
    log.info("mosaic result: %s dtype=%s chunks=%s", da.shape, da.dtype, da.chunksizes)

    log.info("writing COG -> %s", out)
    t0 = time.time()
    da.rio.to_raster(
        out,
        driver="COG",
        dtype="uint8",
        compress="DEFLATE",
        bigtiff="YES",
        overview_resampling="average",
        num_threads="ALL_CPUS",
    )
    log.info("cog written in %.1fs, size %.2f MB", time.time() - t0, out.stat().st_size / 1e6)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", required=True, help="Vantor STAC event slug")
    ap.add_argument("--aoi", required=True, type=Path)
    ap.add_argument("--phase", default="post", choices=["pre", "post"])
    ap.add_argument(
        "--resolution",
        type=float,
        default=0.4 / 111320,
        help="Target GSD in degrees (default ~0.4 m at equator)",
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--asset", default="visual", help="STAC asset key for the RGB COG")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads-per-worker", type=int, default=8)
    ap.add_argument("--memory-limit", default="8GB")
    ap.add_argument("--chunksize", type=int, default=2048)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    aoi = gpd.read_file(args.aoi).to_crs(4326)
    minx, miny, maxx, maxy = map(float, aoi.total_bounds)
    log.info("aoi bbox: %.4f,%.4f,%.4f,%.4f", minx, miny, maxx, maxy)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with (
        LocalCluster(
            n_workers=args.workers, threads_per_worker=args.threads_per_worker, memory_limit=args.memory_limit
        ) as cluster,
        Client(cluster) as client,
    ):
        log.info("dask dashboard: %s", client.dashboard_link)
        items = load_items(args.event, args.phase)
        clouds = [i.properties.get("eo:cloud_cover", 100) for i in items]
        log.info("%d %s scenes; cloud cover %.0f%%..%.0f%%", len(items), args.phase, min(clouds), max(clouds))
        composite(items, (minx, miny, maxx, maxy), args.resolution, args.out, args.chunksize, args.asset)


if __name__ == "__main__":
    main()
