"""Vantor open-data compositor; plan lists per-scene coverage, build writes a least-cloudy composite."""

import argparse
import json
import logging
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from scipy.ndimage import binary_opening
from shapely.geometry import shape as _shape
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vantor-tool")

STAC_BASE = "https://vantor-opendata.s3.amazonaws.com/events"
CLOUD_BRIGHT_T = 200
CLOUD_SAT_T = 20
CLOUD_OPENING_PX = 5


HTTP_TIMEOUT_S = 60


def fetch_collection(event_slug):
    url = f"{STAC_BASE}/{event_slug}/collection.json"
    log.info("fetching collection %s", url)
    return json.loads(urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S).read())


def fetch_item(event_slug, item_id):
    url = f"{STAC_BASE}/{event_slug}/{item_id}.json"
    return json.loads(urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S).read())


def list_items(event_slug, phase="post"):
    coll = fetch_collection(event_slug)
    items = []
    for link in coll.get("links", []):
        if link.get("rel") != "item":
            continue
        item_id = link["href"].split("/")[-1].replace(".json", "")
        item = fetch_item(event_slug, item_id)
        p = item.get("properties", {})
        if p.get("phase") == phase:
            items.append({"id": item_id, "item": item})
    log.info("found %d %s items in %s", len(items), phase.upper(), event_slug)
    return items


def download_scene(item, cache_dir):
    """Download the visual COG to `cache_dir` if not already present. Returns local path as str."""
    cog_url = item["assets"]["visual"]["href"]
    local = cache_dir / f"{item['id']}.tif"
    if local.exists() and local.stat().st_size > 1_000_000:
        return str(local)
    tmp = local.with_suffix(".tif.part")
    log.info("downloading %s -> %s", item["id"], local.name)
    with urllib.request.urlopen(cog_url, timeout=HTTP_TIMEOUT_S) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(local)
    return str(local)


def stream_url(item):
    """Return a rasterio-openable /vsicurl/ URL for the scene's visual COG."""
    return f"/vsicurl/{item['assets']['visual']['href']}"


def scene_source(item, cache_dir, stream):
    return stream_url(item) if stream else download_scene(item, cache_dir)


def real_footprint(tif_path):
    with rasterio.open(tif_path) as src:
        step = max(src.width, src.height) // 1200 or 1
        arr = src.read(1, out_shape=(src.height // step, src.width // step))
        valid = (arr > 0).astype(np.uint8)
        t = src.transform * src.transform.scale(step, step)
        geoms = [_shape(g) for g, v in shapes(valid, mask=valid == 1, transform=t) if v == 1]
    return unary_union(geoms) if geoms else None


def km2(g, crs=4326):
    if g is None or g.is_empty:
        return 0.0
    return gpd.GeoSeries([g], crs=crs).to_crs(3857).area.iloc[0] / 1e6


def plan(event_slug, aoi_path, cache_dir, cloud_cap=100, workers=4, phase="post", stream=True):
    aoi = gpd.read_file(aoi_path).to_crs(4326).geometry.union_all()
    items = list_items(event_slug, phase=phase)
    items.sort(key=lambda x: x["item"]["properties"].get("eo:cloud_cover", 100))

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scene_source, p["item"], cache_dir, stream): p for p in items}
        for f in as_completed(futs):
            p = futs[f]
            src = f.result()
            fp = real_footprint(src)
            inter = fp.intersection(aoi) if fp else None
            props = p["item"]["properties"]
            rows.append(
                {
                    "id": p["id"],
                    "path": src,
                    "footprint": fp,
                    "cloud": props.get("eo:cloud_cover", 100),
                    "off_nadir": props.get("view:off_nadir"),
                    "datetime": props.get("datetime"),
                    "real_km2": km2(fp),
                    "aoi_km2": km2(inter),
                }
            )
    rows.sort(key=lambda r: (r["cloud"], -r["aoi_km2"]))

    aoi_km2 = km2(aoi)
    print()
    print(f"AOI area: {aoi_km2:.2f} km2")
    print(f"{'id':22} {'cloud':>6} {'off_nadir':>10} {'real_km2':>10} {'aoi_km2':>9}  datetime")
    for r in rows:
        print(
            f"{r['id']:22} {r['cloud']:6.1f} {r['off_nadir']!s:>10} "
            f"{r['real_km2']:10.1f} {r['aoi_km2']:9.2f}  {r['datetime']}"
        )

    kept = [r for r in rows if r["cloud"] <= cloud_cap]
    union = unary_union([r["footprint"] for r in kept if r["footprint"]])
    union_aoi = union.intersection(aoi)
    print()
    print(
        f"union of kept ({len(kept)} scenes, cloud<={cloud_cap}%): "
        f"{km2(union):.1f} km2 real  |  {km2(union_aoi):.2f} km2 in AOI "
        f"({km2(union_aoi) / aoi_km2 * 100:.1f}%)"
    )
    return {"aoi": aoi, "scenes": rows, "kept": kept, "union_aoi": union_aoi}


def compute_target_grid(scenes, aoi, buffer_m=200, gsd_m=None):
    union = unary_union([r["footprint"] for r in scenes if r["footprint"]])
    useful = union.intersection(aoi)
    buf_deg = buffer_m / 111000.0
    minx, miny, maxx, maxy = useful.buffer(buf_deg).bounds
    with rasterio.open(scenes[0]["path"]) as r:
        crs = r.crs
        native_xres, native_yres = abs(r.transform.a), abs(r.transform.e)
    if gsd_m is not None:
        # Convert requested metric GSD to degrees using latitude-scaled cosine
        lat_mid = (miny + maxy) / 2.0
        yres = gsd_m / 111000.0
        xres = gsd_m / (111000.0 * np.cos(np.radians(lat_mid)))
        log.info(
            "target GSD: %.2f m/px -> (%.3e, %.3e) deg/px (was native %.3e, %.3e)",
            gsd_m,
            xres,
            yres,
            native_xres,
            native_yres,
        )
    else:
        xres, yres = native_xres, native_yres
        for s in scenes[1:]:
            with rasterio.open(s["path"]) as r:
                xres = min(xres, abs(r.transform.a))
                yres = min(yres, abs(r.transform.e))
    width = int(np.ceil((maxx - minx) / xres))
    height = int(np.ceil((maxy - miny) / yres))
    transform = from_origin(minx, maxy, xres, yres)
    log.info(
        "target grid: %d x %d px @ (%g, %g) deg/px, bounds=(%.4f,%.4f,%.4f,%.4f)",
        width,
        height,
        xres,
        yres,
        minx,
        miny,
        maxx,
        maxy,
    )
    return width, height, transform, crs


def _is_cloud_hard(rgb):
    """Per-pixel: bright and desaturated. Isolated hits from bright roofs are removed by
    a morphological opening in _coherent_cloud_mask before the mask is used."""
    m = rgb.mean(axis=0)
    sat = rgb.max(axis=0).astype(np.int16) - rgb.min(axis=0).astype(np.int16)
    return (m > CLOUD_BRIGHT_T) & (sat < CLOUD_SAT_T)


def _coherent_cloud_mask(rgb, opening_px=CLOUD_OPENING_PX):
    """Drop single-pixel and small-blob hits; only cloud-sized clusters survive."""
    return binary_opening(_is_cloud_hard(rgb), structure=np.ones((opening_px, opening_px), dtype=bool))


def _read_block_with_retry(vrt, win, path, attempts=4):
    """Transient HTTPS/GDAL tile reads occasionally fail mid-block; on failure the whole
    block is retried after a brief backoff. On persistent failure the block is filled with
    zeros so priority-fallback can supply pixels from another scene."""
    for i in range(attempts):
        try:
            return vrt.read([1, 2, 3], window=win)
        except (rasterio.RasterioIOError, rasterio.errors.RasterioIOError) as exc:
            log.warning("read retry %d/%d on %s @ %s: %s", i + 1, attempts, Path(path).name, win, exc)
            time.sleep(0.5 * (i + 1))
    log.warning("read failed permanently on %s @ %s; block filled with zeros", Path(path).name, win)
    return np.zeros((3, int(win.height), int(win.width)), dtype=np.uint8)


def _valid(rgb):
    return rgb.sum(axis=0) > 0


def build_composite(  # noqa: PLR0915  # single macroblock loop with closures; splitting hurts readability
    scenes, aoi, out_dir, buffer_m=200, workers=6, block_px=4096, gsd_m=None, out_prefix="post_composite"
):
    width, height, transform, crs = compute_target_grid(scenes, aoi, buffer_m=buffer_m, gsd_m=gsd_m)
    paths = [s["path"] for s in scenes]
    out_rgb = out_dir / f"{out_prefix}.tif"
    out_src = out_dir / f"{out_prefix}_source.tif"
    out_cloud = out_dir / f"{out_prefix}_cloud.tif"
    profile = dict(
        driver="GTiff",
        count=3,
        dtype="uint8",
        width=width,
        height=height,
        crs=crs,
        transform=transform,
        compress="DEFLATE",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        bigtiff="yes",
        nodata=0,
    )
    one = {**profile, "count": 1}

    tls = threading.local()
    handles: list = []
    handles_lock = threading.Lock()

    def get_vrts():
        # No nodata on the VRT: GDAL rewrites truly-0 source pixels to 1, which stretches map back to 0.
        if not hasattr(tls, "vrts"):
            srcs, vrts = [], []
            for p in paths:
                src = rasterio.open(p)
                vrt = WarpedVRT(
                    src,
                    crs=crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.bilinear,
                )
                srcs.append(src)
                vrts.append(vrt)
            tls.srcs = srcs
            tls.vrts = vrts
            with handles_lock:
                handles.extend(vrts)
                handles.extend(srcs)
        return tls.vrts

    def process(bx, by):
        col = bx * block_px
        row = by * block_px
        w = min(block_px, width - col)
        h = min(block_px, height - row)
        win = Window(col, row, w, h)  # ty: ignore[too-many-positional-arguments]
        vrts = get_vrts()
        reads = [_read_block_with_retry(v, win, paths[i]) for i, v in enumerate(vrts)]
        stack = np.stack(reads, axis=0)
        valid = np.stack([_valid(r) for r in reads])
        cloud = np.stack([_coherent_cloud_mask(r) for r in reads])
        any_valid = valid.any(axis=0)
        all_cloud = (cloud | ~valid).all(axis=0) & any_valid
        # Scenes are pre-sorted by eo:cloud_cover ascending. Assign each pixel to the first
        # valid non-cloudy scene; fall back to the first valid scene where none are clean.
        src_map = np.full((h, w), 255, dtype=np.uint8)
        for si in range(len(reads)):
            pick = (src_map == 255) & valid[si] & ~cloud[si]
            src_map[pick] = si
        for si in range(len(reads)):
            pick = (src_map == 255) & valid[si]
            src_map[pick] = si
        rgb = np.zeros((3, h, w), dtype=np.uint8)
        for si in range(len(reads)):
            m = src_map == si
            for b in range(3):
                rgb[b][m] = stack[si, b][m]
        return bx, by, win, rgb, src_map, all_cloud.astype(np.uint8)

    n_bx = (width + block_px - 1) // block_px
    n_by = (height + block_px - 1) // block_px
    total = n_bx * n_by
    log.info("blocks: %d x %d = %d (workers=%d, sources=%d)", n_bx, n_by, total, workers, len(paths))

    lock = threading.Lock()
    try:
        with (
            rasterio.open(out_rgb, "w", **profile) as dst,
            rasterio.open(out_src, "w", **one) as sd,
            rasterio.open(out_cloud, "w", **one) as cd,
        ):
            t0 = time.time()
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process, bx, by) for by in range(n_by) for bx in range(n_bx)]
                for f in as_completed(futs):
                    _bx, _by, win, rgb, src_map, cloud = f.result()
                    with lock:
                        dst.write(rgb, window=win)
                        sd.write(src_map[None], window=win)
                        cd.write(cloud[None], window=win)
                    done += 1
                    if done % 10 == 0 or done == total:
                        elap = time.time() - t0
                        log.info(
                            "block %d/%d  elapsed=%ds  ~%ds remaining",
                            done,
                            total,
                            int(elap),
                            int(elap / done * (total - done)),
                        )
    finally:
        for h in handles:
            h.close()
    log.info("DONE composite %s (%.1f MB)", out_rgb.name, out_rgb.stat().st_size / 1e6)
    return out_rgb, out_src, out_cloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "build"])
    ap.add_argument("--event", required=True, help="Vantor STAC event slug (e.g. Nepal-Flooding-Aug-2026)")
    ap.add_argument("--aoi", required=True, help="Path to AOI GeoJSON (any CRS)")
    ap.add_argument("--cache-dir", default="./vantor_cache", help="Where to store downloaded COGs")
    ap.add_argument("--out-dir", default="./vantor_composite", help="Where to write composite (build mode)")
    ap.add_argument("--cloud-cap", type=float, default=100.0, help="Skip scenes above this cloud percent")
    ap.add_argument("--buffer-m", type=float, default=200.0)
    ap.add_argument(
        "--gsd-m",
        type=float,
        default=None,
        help="Target GSD in meters/pixel (default: finest native across scenes; huge for wide bboxes)",
    )
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--block-px", type=int, default=4096)
    ap.add_argument(
        "--phase",
        choices=["pre", "post"],
        default="post",
        help="Which STAC phase to composite (default post).",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download each scene to --cache-dir first. Default streams via /vsicurl/.",
    )
    args = ap.parse_args()

    stream = not args.download
    cache_dir = Path(args.cache_dir)
    if not stream:
        cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdal_env = {"GDAL_HTTP_MAX_RETRY": "3", "GDAL_HTTP_RETRY_DELAY": "1"} if stream else {}
    with rasterio.Env(**gdal_env):
        result = plan(
            args.event,
            args.aoi,
            cache_dir,
            cloud_cap=args.cloud_cap,
            workers=args.workers,
            phase=args.phase,
            stream=stream,
        )
        if args.mode == "build":
            build_composite(
                result["kept"],
                result["aoi"],
                out_dir,
                buffer_m=args.buffer_m,
                workers=args.workers,
                block_px=args.block_px,
                gsd_m=args.gsd_m,
                out_prefix=f"{args.phase}_composite",
            )


if __name__ == "__main__":
    sys.exit(main())
