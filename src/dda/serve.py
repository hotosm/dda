import asyncio
import glob
import json
import logging
import os
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer

import geopandas as gpd
import rasterio
from rasterio.merge import merge

from dda.config import load_config
from dda.infer import load_model, resolve_ckpt, sliding_window_prob
from dda.pool import (
    assign_damage,
    filter_by_confidence,
    mark_uncovered_as_no_data,
    parse_confidence_threshold,
)
from dda.tiling import imaged_mask, read_aligned_pair

log = logging.getLogger(__name__)

OSM_API_URL = os.environ.get("DDA_OSM_API_URL", "https://api-prod.raw-data.hotosm.org/v1")
OSM_BUILDING_FILTER = {
    "tags": {
        "polygon": {
            "join_or": {
                "building": [],
                "destroyed:building": [],
                "damaged:building": [],
                "damage:building": [],
            }
        }
    }
}

# Named like the inputs of the `dda export` ONNX model.
IMAGERY_INPUTS = ("post", "pre")
CFG = load_config(None)
DEVICE = os.environ.get("DDA_DEVICE") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")


def _resolve_ckpt() -> str:
    ref = os.environ.get("DDA_DAMAGE_CKPT")
    if ref and Path(ref).exists():
        return ref
    if ref and ":" in ref:
        from huggingface_hub import hf_hub_download

        repo_id, filename = ref.split(":", 1)
        return hf_hub_download(repo_id=repo_id, filename=filename)
    return resolve_ckpt(CFG, ref)


MODEL = load_model(_resolve_ckpt(), CFG, device=DEVICE)
log.info("damage model loaded on %s", DEVICE)


def _download_raster(tms: str, bbox: list[float], zoom: int, out_dir: str, name: str) -> str:
    from geomltoolkits.downloader.tms import download_tiles

    tiles_out = str(Path(out_dir) / name)
    asyncio.run(download_tiles(tms=tms, zoom=zoom, out=tiles_out, bbox=bbox, georeference=True))
    tifs = glob.glob(f"{tiles_out}/**/*.tif", recursive=True)
    if not tifs:
        raise RuntimeError(f"no tiles downloaded for {name} from {tms}")
    datasets = [rasterio.open(t) for t in tifs]
    mosaic, transform = merge(datasets)
    meta = datasets[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=transform, count=mosaic.shape[0])
    raster_path = str(Path(out_dir) / f"{name}.tif")
    with rasterio.open(raster_path, "w", **meta) as dst:
        dst.write(mosaic)
    for d in datasets:
        d.close()
    return raster_path


def _fetch_osm_buildings(bbox: list[float], out_dir: str) -> str:
    from geomltoolkits.downloader.osm import download_osm_data

    osm_dir = Path(out_dir) / "osm"
    osm_dir.mkdir(exist_ok=True)
    asyncio.run(
        download_osm_data(
            bbox=bbox, api_url=OSM_API_URL, filters=OSM_BUILDING_FILTER, dump_results=True, out=str(osm_dir)
        )
    )
    result = osm_dir / "osm-result.geojson"
    if not result.exists():
        raise RuntimeError("OSM building download produced no file")
    return str(result)


def _predict(imagery: dict[str, str], bbox: list[float], zoom: int, confidence_threshold: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="dda-serve-") as tmp:
        post_raster = _download_raster(imagery["post"], bbox, zoom, tmp, "post")
        pre_raster = _download_raster(imagery["pre"], bbox, zoom, tmp, "pre")
        buildings_geojson = _fetch_osm_buildings(bbox, tmp)
        post, pre, transform, crs = read_aligned_pair(CFG, post_raster, pre_raster)

        prob = sliding_window_prob(
            MODEL, post, pre, CFG.tile_window, CFG.tile_stride, CFG.temperature, DEVICE
        )
        buildings = gpd.read_file(buildings_geojson)
        gdf = assign_damage(
            prob, transform, crs, buildings, pool_op=CFG.pool_op, percentile=CFG.pool_percentile
        )
        gdf = mark_uncovered_as_no_data(gdf, imaged_mask(post, pre), transform)
        gdf = filter_by_confidence(gdf, confidence_threshold)
        keep = ["geometry", "damage", "damage_class", "damage_confidence"]
        keep += [c for c in ("osm_id", "id") if c in gdf.columns]
        return json.loads(gdf[keep].to_crs(4326).to_json())


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "device": DEVICE})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/predict":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
            imagery = payload.get("imagery")
            if not isinstance(imagery, dict) or not all(
                isinstance(imagery.get(name), str) and imagery[name] for name in IMAGERY_INPUTS
            ):
                raise ValueError('imagery requires TMS templates for post and pre: {"post": ..., "pre": ...}')
            bbox = [float(v) for v in payload["bbox"]]
            if len(bbox) != 4:
                raise ValueError("bbox needs 4 numbers: west, south, east, north")
            zoom = int(payload["zoom"])
            params = payload.get("params") or {}
            if not isinstance(params, dict):
                raise TypeError("params must be an object")
            confidence_threshold = parse_confidence_threshold(params.get("confidence_threshold", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            self._json(400, {"error": f"invalid request: {exc}"})
            return
        result = _predict(imagery, bbox, zoom, confidence_threshold)
        self._json(200, result)


class Server(ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.environ.get("PORT", "8080"))
    with Server(("0.0.0.0", port), Handler) as server:
        log.info("dda damage serving on :%d", port)
        server.serve_forever()


if __name__ == "__main__":
    main()
