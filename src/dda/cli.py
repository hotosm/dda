"""dda CLI.

Two command groups:

Per-area pipeline (the slim disaster-assessment path):
  prepare              fetch pre + post rasters, coreg + photometric calibration
  buildings            detect footprints (fAIr) OR pull from OSM (--source osm) OR use your own (--input)
  damage               score each building; writes damage.geojson
  eval                 per-class F1 / confusion matrix vs a labelled ground-truth GeoJSON
  fewshot buildings    backbone-frozen fine-tune of fAIr (from TM projects OR local chips+labels)
  fewshot damage       fine-tune the damage model (strict head-only by default; --full-decoder for more)
  publish              upload the area's damage.geojson to a HF dataset

Training / dev-time (existing):
  train  predict  export  evaluate  calibrate
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dda.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dda", description="DINOv3 siamese building damage assessment")
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", default="conf/train.yaml", help="Path to YAML config")
        p.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. data_pct=10")

    _common(sub.add_parser("train", help="Train the damage model"))

    p_predict = sub.add_parser("predict", help="Damage prediction on a post GeoTIFF + building GeoJSON")
    p_predict.add_argument("--ckpt", default=None)
    p_predict.add_argument("--raster", required=True, help="Post-disaster GeoTIFF")
    p_predict.add_argument("--buildings", required=True, help="Building footprints GeoJSON")
    p_predict.add_argument("--pre-raster", default=None, help="Optional pre-disaster GeoTIFF")
    p_predict.add_argument("--out", required=True)
    _common(p_predict)

    p_export = sub.add_parser("export", help="Export checkpoint to ONNX")
    p_export.add_argument("--ckpt", default=None)
    p_export.add_argument("--out", required=True)
    _common(p_export)

    p_evaluate = sub.add_parser("evaluate", help="Object-level (per-building) F1 on xBD splits")
    p_evaluate.add_argument("--ckpt", default=None)
    p_evaluate.add_argument("--split", default="val", choices=["val", "test"])
    _common(p_evaluate)

    p_cal = sub.add_parser("calibrate", help="Fit confidence temperature on the val split")
    p_cal.add_argument("--ckpt", default=None)
    _common(p_cal)

    sub.add_parser("serve", help="Run the damage inference HTTP server (POST /predict)")

    _add_pipeline_parsers(sub)
    return parser


def _add_pipeline_parsers(sub) -> None:  # single dispatch table; splitting hurts readability
    p_prep = sub.add_parser("prepare", help="Fetch pre + post rasters, coreg + photometric calibration")
    p_prep.add_argument("--area", required=True)
    p_prep.add_argument("--aoi", required=True)
    p_prep.add_argument(
        "--pre-img",
        required=True,
        help="Pre imagery source: TMS URL, COG URL, or bing hostname",
    )
    p_prep.add_argument(
        "--post-img",
        required=True,
        help="Post imagery source: TMS URL, COG URL, or bing hostname",
    )
    p_prep.add_argument("--zoom", type=int, default=19, help="TMS zoom level; ignored for COGs")
    p_prep.add_argument("--outputs-root", default="outputs")
    p_prep.add_argument(
        "--no-photometric-calibration",
        action="store_true",
        help="Skip the per-band mean/std match at coreg time (default: on)",
    )
    p_prep.add_argument(
        "--no-stretch",
        action="store_true",
        help="Skip the 2-98 percentile stretch on pre_aligned + post (default: on)",
    )
    p_prep.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw pre.tif + post.tif after prepare (default: deleted to save disk)",
    )

    p_bld = sub.add_parser(
        "buildings",
        help="Building footprints: fAIr detection, OSM via PostPass, or user-supplied GeoJSON",
    )
    p_bld.add_argument("--area", required=True)
    p_bld.add_argument("--outputs-root", default="outputs")
    p_bld.add_argument(
        "--source",
        default="fair",
        choices=["fair", "osm"],
        help="fair = DINOv3-S UperNet inference on pre_aligned; osm = pull building footprints from OSM",
    )
    p_bld.add_argument(
        "--osm-source",
        default="raw_data_api",
        choices=["raw_data_api", "postpass"],
        help="Backend for --source osm; raw_data_api supports lifecycle tag families (destroyed:building)",
    )
    p_bld.add_argument(
        "--input",
        default=None,
        help="Skip detection; copy this GeoJSON as the buildings.geojson (must have Polygon geometries)",
    )
    p_bld.add_argument(
        "--ckpt",
        default=None,
        help="Fine-tuned buildings ckpt path (e.g. from `dda fewshot buildings`); defaults to pretrained",
    )
    p_bld.add_argument(
        "--config",
        "-c",
        default=None,
        help="Optional TrainConfig YAML; used to source osm_tag_families for --source osm",
    )

    p_dam = sub.add_parser(
        "damage",
        help="Score buildings for damage; writes damage.geojson",
    )
    p_dam.add_argument("--area", required=True)
    p_dam.add_argument("--ckpt", default=None)
    p_dam.add_argument("--outputs-root", default="outputs")
    p_dam.add_argument("--config", "-c", default="conf/train.yaml")
    p_dam.add_argument("overrides", nargs="*")

    p_eval = sub.add_parser("eval", help="Per-class F1 / confusion matrix vs labelled ground truth")
    p_eval.add_argument("--predictions", required=True, help="Path to a damage.geojson")
    p_eval.add_argument("--labels", required=True, help="Ground-truth GeoJSON with a `damage` column")
    p_eval.add_argument("--out", default=None, help="Optional path to save the metrics as JSON")

    _add_fewshot_parsers(sub)

    p_pub = sub.add_parser("publish", help="Upload the area's damage.geojson to a HF dataset")
    p_pub.add_argument("--area", required=True)
    p_pub.add_argument("--repo-id", required=True)
    p_pub.add_argument("--outputs-root", default="outputs")

    p_run = sub.add_parser(
        "run",
        help="Run the whole pipeline from one YAML: aoi/prepare/fewshot/buildings/damage/publish",
    )
    p_run.add_argument("--config", required=True, help="Path to an event YAML (see conf/event.example.yaml)")
    p_run.add_argument(
        "-s",
        "--start",
        default=None,
        choices=["aoi", "prepare", "fewshot", "buildings", "damage", "publish"],
        help="Start from this stage and run through the end",
    )
    p_run.add_argument(
        "--only",
        default=None,
        choices=["aoi", "prepare", "fewshot", "buildings", "damage", "publish"],
        help="Run exactly this one stage",
    )
    p_run.add_argument("--dry-run", action="store_true", help="Print stage plan without executing")
    p_run.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. buildings.fewshot.hpo_trials=4",
    )


def _add_fewshot_parsers(sub) -> None:
    p_fs = sub.add_parser("fewshot", help="Quickly fit a model to a new area")
    fs_sub = p_fs.add_subparsers(dest="fewshot_kind", required=True)

    p_fs_b = fs_sub.add_parser(
        "buildings",
        help="Backbone-frozen fine-tune of fAIr from TM projects OR local chips+labels",
    )
    p_fs_b.add_argument(
        "--tm-projects",
        default=None,
        help="Comma-separated HOT Tasking Manager project IDs, e.g. `1234,1235`",
    )
    p_fs_b.add_argument(
        "--chips", default=None, help="Directory of georeferenced RGB .tif chips (manual path)"
    )
    p_fs_b.add_argument("--labels", default=None, help="Building-polygon GeoJSON in EPSG:4326 (manual path)")
    p_fs_b.add_argument("--imagery-tms", default=None, help="Override TM's `imagery` URL (needed when null)")
    p_fs_b.add_argument("--zoom", type=int, default=19, help="TMS zoom level (TM path only)")
    p_fs_b.add_argument("--out-dir", required=True, help="Output directory: chips + masks + ckpts + summary")
    p_fs_b.add_argument("--epochs", type=int, default=10)
    p_fs_b.add_argument("--lr", type=float, default=5e-5, help="Ignored when HPO is on")
    p_fs_b.add_argument("--val-frac", type=float, default=0.3)
    p_fs_b.add_argument("--patience", type=int, default=3)
    p_fs_b.add_argument(
        "--hpo-trials",
        type=int,
        default=8,
        help="Optuna trials over (lr, weight_decay). Default 8. Set 0 for single manual fit.",
    )
    p_fs_b.add_argument(
        "--hpo-seeds",
        type=int,
        default=1,
        help="Random-split seeds averaged per HPO trial. 1 = fast, 3 = robust (removes split noise)",
    )
    p_fs_b.add_argument("--no-hpo", action="store_true", help="Shortcut for --hpo-trials 0")

    p_fs_d = fs_sub.add_parser(
        "damage",
        help="Fine-tune the damage model on one area (default: only classifier heads move)",
    )
    p_fs_d.add_argument("--pre", required=True, help="Pre-disaster raster (GeoTIFF, any CRS)")
    p_fs_d.add_argument("--post", required=True, help="Post-disaster raster (GeoTIFF; defines chip grid)")
    p_fs_d.add_argument(
        "--labels",
        required=True,
        help="GeoJSON of labelled buildings with a `damage` column (int 1..4 or string names)",
    )
    p_fs_d.add_argument("--out-dir", required=True, help="Output directory for ckpts + logs")
    p_fs_d.add_argument("--ckpt", default=None, help="Starting checkpoint (auto-download if omitted)")
    p_fs_d.add_argument("--epochs", type=int, default=10)
    p_fs_d.add_argument("--lr", type=float, default=1e-4)
    p_fs_d.add_argument("--tile-size", type=int, default=512)
    p_fs_d.add_argument("--stride", type=int, default=384)
    p_fs_d.add_argument("--val-frac", type=float, default=0.3)
    p_fs_d.add_argument("--patience", type=int, default=3)
    p_fs_d.add_argument("--batch-size", type=int, default=2)
    p_fs_d.add_argument(
        "--full-decoder",
        action="store_true",
        help="Unfreeze the pyramid + UperNet decoder + fusion (default: strict head-only)",
    )


def app() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "serve":
        from dda.serve import main as serve_main

        serve_main()
        return 0
    if args.command in {"train", "predict", "export", "evaluate", "calibrate"}:
        return _run_dev(args)
    return _run_pipeline(args)


def _run_dev(args) -> int:
    cfg = load_config(args.config, overrides=list(args.overrides))
    if args.command == "train":
        from dda.train import train

        print(json.dumps(train(cfg), indent=2))
    elif args.command == "predict":
        from dda.infer import predict_damage, resolve_ckpt

        predict_damage(
            cfg,
            ckpt_path=resolve_ckpt(cfg, args.ckpt),
            post_raster=args.raster,
            buildings_geojson=args.buildings,
            out_geojson=args.out,
            pre_raster=args.pre_raster,
        )
    elif args.command == "export":
        from dda.export import export_onnx
        from dda.infer import resolve_ckpt

        export_onnx(cfg, ckpt_path=resolve_ckpt(cfg, args.ckpt), out_path=args.out)
    elif args.command == "evaluate":
        from dda.evaluation import object_level_eval
        from dda.infer import resolve_ckpt

        result = object_level_eval(cfg, ckpt_path=resolve_ckpt(cfg, args.ckpt), split=args.split)
        print(json.dumps(result, indent=2))
    elif args.command == "calibrate":
        from dda.calibrate import fit_temperature
        from dda.infer import resolve_ckpt

        temperature = fit_temperature(cfg, ckpt_path=resolve_ckpt(cfg, args.ckpt))
        print(json.dumps({"temperature": temperature}, indent=2))
    return 0


def _run_pipeline(args) -> int:  # noqa: PLR0911  # dispatch table by args.command
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "fewshot":
        return _run_fewshot(args)
    if args.command == "run":
        from dda.event_config import load_event_config
        from dda.pipeline.runner import run_event

        cfg = load_event_config(args.config, overrides=list(args.overrides))
        run_event(cfg, start=args.start, only=args.only, dry_run=args.dry_run)
        return 0

    from dda.pipeline.paths import PipelinePaths

    paths = PipelinePaths.for_area(args.area, outputs_root=getattr(args, "outputs_root", "outputs"))
    paths.ensure_dirs()
    if args.command == "prepare":
        return _run_prepare(args, paths)
    if args.command == "buildings":
        return _run_buildings(args, paths)
    if args.command == "damage":
        return _run_damage(args, paths)
    if args.command == "publish":
        return _run_publish(args, paths)
    raise SystemExit(f"unknown command: {args.command}")


def _run_prepare(args, paths) -> int:
    from shutil import copy2

    from dda.pipeline.coreg import coregister
    from dda.pipeline.fetch import fetch_raster

    src = Path(args.aoi).resolve()
    dst = paths.aoi.resolve()
    paths.aoi.parent.mkdir(parents=True, exist_ok=True)
    if src != dst:
        copy2(src, dst)
    fetch_raster(args.pre_img, paths.aoi, paths.pre_raw, zoom=args.zoom)
    fetch_raster(args.post_img, paths.aoi, paths.post_raw, zoom=args.zoom)
    coregister(
        paths.pre_raw,
        paths.post_raw,
        paths.pre_aligned,
        paths.post_aligned,
        paths.drift_json,
        paths.coreg_check_png,
        calibrate_photometry=not args.no_photometric_calibration,
        stretch_percentiles=not args.no_stretch,
        keep_raw=args.keep_raw,
    )
    return 0


def _run_buildings(args, paths) -> int:
    if args.input:
        import geopandas as gpd

        from dda.pipeline.geowrite import write_dual

        gdf = gpd.read_file(args.input).to_crs("EPSG:4326").reset_index(drop=True)
        if "score" in gdf.columns and "building_confidence" not in gdf.columns:
            gdf = gdf.rename(columns={"score": "building_confidence"})
        if "building_confidence" not in gdf.columns:
            gdf["building_confidence"] = 1.0
        if "source" not in gdf.columns:
            gdf["source"] = "user"
        if "id" not in gdf.columns:
            gdf["id"] = range(len(gdf))
        write_dual(gdf[["id", "building_confidence", "source", "geometry"]], paths.buildings)
        logging.getLogger(__name__).info(
            "buildings: copied %d user-supplied footprints -> %s(.geojson|.parquet)",
            len(gdf),
            paths.buildings.with_suffix(""),
        )
        return 0
    if args.source == "osm":
        if args.osm_source == "postpass":
            from dda.pipeline.postpass import fetch_postpass_buildings

            fetch_postpass_buildings(paths.aoi, paths.buildings)
            return 0
        import geopandas as gpd

        from dda.pipeline.geowrite import write_dual
        from dda.pipeline.osm_pull import pull_osm_buildings

        cfg = load_config(args.config) if args.config else load_config(None)
        families = [dict(f) for f in cfg.osm_tag_families]
        gdf = pull_osm_buildings(
            aoi_geojson=paths.aoi,
            cache_dir=paths.root / "osm_cache",
            families=families,
        )
        write_dual(gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs), paths.buildings)
        return 0
    from dda.pipeline.buildings import run_fair_buildings

    run_fair_buildings(
        paths.pre_aligned,
        paths.aoi,
        paths.buildings,
        ckpt=Path(args.ckpt) if args.ckpt else None,
    )
    return 0


def _run_damage(args, paths) -> int:
    from dda.infer import resolve_ckpt
    from dda.pipeline.damage import run_damage_blocked

    cfg = load_config(args.config, overrides=list(args.overrides))
    run_damage_blocked(
        cfg=cfg,
        ckpt_path=resolve_ckpt(cfg, args.ckpt),
        post_raster=paths.post_aligned,
        pre_aligned=paths.pre_aligned,
        buildings_geojson=paths.buildings,
        out_geojson=paths.damage,
    )
    return 0


def _run_eval(args) -> int:
    from dda.pipeline.eval_preds import evaluate

    result = evaluate(
        predictions=Path(args.predictions),
        labels=Path(args.labels),
        out_json=Path(args.out) if args.out else None,
    )
    print(json.dumps(result, indent=2))
    return 0


def _run_fewshot(args) -> int:
    if args.fewshot_kind == "buildings":
        return _run_fewshot_buildings(args)
    if args.fewshot_kind == "damage":
        from dda.pipeline.fewshot import fit_damage

        result = fit_damage(
            pre_raster=Path(args.pre),
            post_raster=Path(args.post),
            labels_geojson=Path(args.labels),
            out_dir=Path(args.out_dir),
            ckpt=Path(args.ckpt) if args.ckpt else None,
            epochs=args.epochs,
            lr=args.lr,
            tile_size=args.tile_size,
            stride=args.stride,
            val_frac=args.val_frac,
            patience=args.patience,
            batch_size=args.batch_size,
            strict_head=not args.full_decoder,
        )
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    raise SystemExit(f"unknown fewshot kind: {args.fewshot_kind}")


def _run_fewshot_buildings(args) -> int:
    from dda.pipeline.fewshot import fit_buildings, fit_buildings_from_tm

    out_dir = Path(args.out_dir)
    if args.tm_projects:
        pids = [int(x) for x in args.tm_projects.split(",") if x.strip()]
        if not pids:
            raise SystemExit("--tm-projects must be a non-empty comma-separated list of integers")
        if args.chips or args.labels:
            raise SystemExit("--tm-projects is mutually exclusive with --chips / --labels; pick one path")
        hpo_trials = 0 if args.no_hpo else args.hpo_trials
        result = fit_buildings_from_tm(
            project_ids=pids,
            out_dir=out_dir,
            zoom=args.zoom,
            imagery_tms_override=args.imagery_tms,
            epochs=args.epochs,
            lr=args.lr,
            val_frac=args.val_frac,
            patience=args.patience,
            hpo_trials=hpo_trials,
            hpo_seeds=args.hpo_seeds,
        )
    else:
        if not (args.chips and args.labels):
            raise SystemExit("manual path requires both --chips and --labels (or use --tm-projects)")
        result = fit_buildings(
            chips_dir=Path(args.chips),
            labels_geojson=Path(args.labels),
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            val_frac=args.val_frac,
            patience=args.patience,
        )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _run_publish(args, paths) -> int:
    from dda.pipeline.publish import push_area_to_hf

    if not paths.damage.exists():
        raise FileNotFoundError(
            f"damage.geojson missing at {paths.damage}. Run `dda damage --area {args.area}` first."
        )
    push_area_to_hf(area=args.area, area_dir=paths.root, repo_id=args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(app())
