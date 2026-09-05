"""HOT raw-data-api snapshot client; one task per tag family, `attributes` stays unset so osm_id survives."""

import io
import json
import logging
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

log = logging.getLogger(__name__)

RDA_BASE = "https://api-prod.raw-data.hotosm.org/v1"
USER_AGENT = "dda-pipeline/0.1 (contact: krschap@duck.com)"
POLL_INTERVAL_S = 5
POLL_TIMEOUT_S = 900

OUTPUT_COLUMNS = ("osm_id", "osm_type", "building", "osm_status", "geometry")


def pull_osm_buildings(
    aoi_geojson: Path,
    cache_dir: Path,
    families: list[dict[str, str]],
) -> gpd.GeoDataFrame:
    """One RDA snapshot per family; merged, deduped, tagged with osm_status."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    aoi_geom = json.loads(Path(aoi_geojson).read_text())

    frames: list[gpd.GeoDataFrame] = []
    for fam in families:
        key = fam["key"]
        status = fam["status"]
        zip_path = cache_dir / f"rda_{key.replace(':', '_')}.zip"
        if not zip_path.exists():
            _fetch_snapshot(aoi_geom, key, zip_path)
        gdf = _extract_geojson(zip_path)
        if len(gdf) == 0:
            log.info("rda family %s: 0 features", key)
            continue
        frames.append(_shape_family_frame(gdf, key=key, status=status))
        log.info("rda family %s (%s): %d features", key, status, len(frames[-1]))

    if not frames:
        empty = {c: [] for c in OUTPUT_COLUMNS}
        return gpd.GeoDataFrame(empty, geometry="geometry", crs="EPSG:4326")

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
    merged["_rank"] = merged["osm_status"].map({"destroyed": 0, "standing": 1}).fillna(2)
    merged = merged.sort_values("_rank").drop_duplicates(subset=["osm_id", "osm_type"], keep="first")
    merged = merged.drop(columns=["_rank"]).reset_index(drop=True)
    return merged[list(OUTPUT_COLUMNS)]


def _fetch_snapshot(aoi_geom: dict[str, Any], key: str, zip_out: Path) -> None:
    payload = {
        "geometry": aoi_geom,
        "geometryType": ["polygon"],
        "filters": {"tags": {"polygon": {"join_or": {key: []}}}},
        "outputType": "geojson",
    }
    req = urllib.request.Request(
        f"{RDA_BASE}/snapshot/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        submit = json.loads(resp.read())
    task_id = submit["task_id"]
    log.info("rda submitted %s -> task %s", key, task_id)
    download_url = _poll_task(task_id)
    with urllib.request.urlopen(download_url, timeout=180) as resp:
        zip_out.write_bytes(resp.read())


def _poll_task(task_id: str) -> str:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    status_url = f"{RDA_BASE}/tasks/status/{task_id}/"
    while True:
        with urllib.request.urlopen(status_url, timeout=30) as resp:
            body = json.loads(resp.read())
        state = body.get("status")
        if state == "SUCCESS":
            url = (body.get("result") or {}).get("download_url")
            if not url:
                raise RuntimeError(f"rda task {task_id} SUCCESS without download_url: {body}")
            return url
        if state == "FAILURE":
            raise RuntimeError(f"rda task {task_id} FAILURE: {body}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"rda task {task_id} not done after {POLL_TIMEOUT_S}s")
        time.sleep(POLL_INTERVAL_S)


def _extract_geojson(zip_path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".geojson")]
        if not names:
            raise RuntimeError(f"rda zip {zip_path} contains no .geojson")
        with zf.open(names[0]) as fp:
            return gpd.read_file(io.BytesIO(fp.read()))


def _shape_family_frame(gdf: gpd.GeoDataFrame, *, key: str, status: str) -> gpd.GeoDataFrame:
    if "osm_id" not in gdf.columns:
        raise RuntimeError(f"rda output for {key} missing osm_id; got {gdf.columns.tolist()}")
    if key not in gdf.columns:
        cols = gdf.columns.tolist()
        raise RuntimeError(f"rda output for {key} missing the requested tag column; got {cols}")
    verified = gdf[gdf[key].notna() & (gdf[key].astype(str).str.strip() != "")].copy()
    dropped = len(gdf) - len(verified)
    if dropped:
        log.info("rda family %s: dropped %d rows with empty %s tag", key, dropped, key)
    out = verified[["osm_id", "geometry"]].copy()
    out["osm_type"] = verified["osm_type"] if "osm_type" in verified.columns else "ways_poly"
    out["building"] = verified[key].astype(str)
    out["osm_status"] = status
    return out[list(OUTPUT_COLUMNS)].to_crs("EPSG:4326")
