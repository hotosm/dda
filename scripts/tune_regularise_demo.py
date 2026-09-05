"""Smoke-test the RegulariseParams TPE tuner on a prepared AOI against OSM buildings."""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd

from dda.pipeline.postpass import fetch_postpass_buildings
from dda.pipeline.tune_regularise import tune_regularise

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aoi-dir",
        required=True,
        type=Path,
        help="Prepared AOI directory containing buildings.geojson, aoi.geojson, pre_aligned.tif",
    )
    parser.add_argument("--raw", type=Path, default=None, help="Overrides <aoi_dir>/buildings.geojson")
    parser.add_argument("--aoi", type=Path, default=None, help="Overrides <aoi_dir>/aoi.geojson")
    parser.add_argument("--pre-raster", type=Path, default=None, help="Overrides <aoi_dir>/pre_aligned.tif")
    parser.add_argument("--gt", type=Path, default=None, help="Overrides <aoi_dir>/osm_buildings_gt.geojson")
    parser.add_argument("--out", type=Path, default=None, help="Overrides <aoi_dir>/regularise_tuned.json")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    aoi_dir = args.aoi_dir
    raw_path = args.raw or aoi_dir / "buildings.geojson"
    aoi_path = args.aoi or aoi_dir / "aoi.geojson"
    pre_raster = args.pre_raster or aoi_dir / "pre_aligned.tif"
    gt_path = args.gt or aoi_dir / "osm_buildings_gt.geojson"
    out_json = args.out or aoi_dir / "regularise_tuned.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw predictions: {raw_path}")
    if not aoi_path.exists():
        raise FileNotFoundError(f"missing AOI: {aoi_path}")
    if not gt_path.exists():
        log.info("no cached OSM GT; fetching via PostPass -> %s", gt_path)
        fetch_postpass_buildings(aoi_path, gt_path)

    raw = gpd.read_file(raw_path).to_crs("EPSG:4326")
    gt = gpd.read_file(gt_path).to_crs("EPSG:4326")
    log.info("loaded raw=%d gt=%d polygons", len(raw), len(gt))

    log.info("starting Optuna TPE: n_trials=%d seed=%d", args.n_trials, args.seed)
    best_params, report = tune_regularise(
        raw_gdf=raw,
        gt_gdf=gt,
        raster_path=str(pre_raster) if pre_raster.exists() else None,
        n_trials=args.n_trials,
        seed=args.seed,
    )
    log.info(
        "best_value=%.4f trials=%d elapsed=%.1fs",
        report["best_value"],
        report["n_trials"],
        report["elapsed_s"],
    )

    out_json.write_text(
        json.dumps(
            {
                "best_value": report["best_value"],
                "n_trials": report["n_trials"],
                "elapsed_s": report["elapsed_s"],
                "best_params": asdict(best_params),
            },
            indent=2,
        )
    )
    log.info("wrote %s", out_json)


if __name__ == "__main__":
    main()
