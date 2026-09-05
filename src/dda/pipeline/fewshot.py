"""Few-shot fit of the fAIr buildings model with the backbone frozen; damage variant re-exported here."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dda.pipeline.fewshot_damage import FewshotDamageResult, fit_damage

log = logging.getLogger(__name__)


__all__ = [
    "FewshotBuildingsResult",
    "FewshotDamageResult",
    "HpoResult",
    "fit_buildings",
    "fit_buildings_from_tm",
    "fit_damage",
    "hpo_buildings",
]

TINY_DATASET_WARN_THRESHOLD = 20
HPO_LR_RANGE = (1e-5, 5e-4)
HPO_WD_RANGE = (1e-4, 1e-2)
HPO_DEFAULT_TRIALS = 8
HPO_DEFAULT_SEEDS = 1


@dataclass(frozen=True)
class FewshotBuildingsResult:
    n_train: int
    n_val: int
    val_iou_pretrained: float
    val_iou_finetuned: float
    delta: float
    best_ckpt: Path
    output_dir: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "val_iou_pretrained": self.val_iou_pretrained,
            "val_iou_finetuned": self.val_iou_finetuned,
            "delta": self.delta,
            "best_ckpt": str(self.best_ckpt),
            "output_dir": str(self.output_dir),
        }


def _stretch_chip_dir(chips_dir: Path, low: float = 2.0, high: float = 98.0, prefix: str = "") -> None:
    """Pooled per-band [low, high] percentile stretch over every `{prefix}*.tif` in `chips_dir`."""
    import numpy as np
    import rasterio

    tifs = sorted(p for p in chips_dir.glob(f"{prefix}*.tif") if not p.name.endswith(".aux.xml"))
    if not tifs:
        return
    sample_stride = 4
    pooled: list[np.ndarray] = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            arr = src.read([1, 2, 3])[:, ::sample_stride, ::sample_stride]
        valid = arr.sum(axis=0) > 0
        if not valid.any():
            continue
        pooled.append(arr[:, valid])
    if not pooled:
        log.warning("stretch: no valid pixels across %d chips; skipping", len(tifs))
        return
    all_pixels = np.concatenate(pooled, axis=1)
    lows, highs = [], []
    for b in range(3):
        lo, hi = np.percentile(all_pixels[b].astype(np.float32), [low, high])
        if hi - lo < 1.0:
            log.warning(
                "stretch: band %d degenerate percentile range (lo=%.2f hi=%.2f); widening to lo+1",
                b,
                float(lo),
                float(hi),
            )
            hi = lo + 1.0
        lows.append(float(lo))
        highs.append(float(hi))
    log.info(
        "stretch: pooled %d chips, per-band lows=%s highs=%s",
        len(tifs),
        [round(x, 1) for x in lows],
        [round(x, 1) for x in highs],
    )
    min_spread = 5.0
    if all((h - lo) < min_spread for lo, h in zip(lows, highs, strict=True)):
        raise RuntimeError(
            f"pooled chip 2-98 spread < {min_spread} on every band "
            f"(lows={[round(x, 1) for x in lows]}, highs={[round(x, 1) for x in highs]}); "
            f"TMS likely lacks coverage at this zoom, try a coarser one."
        )
    for tif in tifs:
        with rasterio.open(tif) as src:
            arr = src.read([1, 2, 3]).astype(np.float32)
            profile = src.profile
        nodata = arr.sum(axis=0) == 0
        out = np.empty_like(arr, dtype=np.uint8)
        for b in range(3):
            scaled = (arr[b] - lows[b]) * 255.0 / (highs[b] - lows[b])
            out[b] = np.clip(scaled, 0, 255).astype(np.uint8)
        out[:, nodata] = 0
        with rasterio.open(tif, "w", **profile) as dst:
            dst.write(out)


def _buildings_config():
    """dinov3_hot ViT-L config with batch 8 so OneCycleLR has enough steps; fp32 off CUDA."""
    import torch
    from dinov3_hot.config import load_config

    cfg = load_config(None)
    cfg.batch_size = 8
    cfg.eval_batch_size = 8
    if not torch.cuda.is_available():
        cfg.precision = "32"
    return cfg


def fit_buildings(
    chips_dir: Path,
    labels_geojson: Path,
    out_dir: Path,
    epochs: int = 10,
    lr: float = 5e-5,
    val_frac: float = 0.3,
    patience: int = 3,
) -> FewshotBuildingsResult:
    """Backbone-frozen fine-tune of dinov3l-buildings on `chips_dir` + `labels_geojson`.

    Chips must be georeferenced RGB `.tif`.
    """
    from dinov3_hot.finetune import finetune

    from dda.pipeline.buildings import _resolve_building_ckpt
    from dda.pipeline.hub_utils import enable_offline_torch_hub_fallback

    enable_offline_torch_hub_fallback()

    chips_dir = Path(chips_dir).resolve()
    labels_geojson = Path(labels_geojson).resolve()
    out_dir = Path(out_dir).resolve()
    if not chips_dir.is_dir():
        raise FileNotFoundError(f"chips_dir does not exist: {chips_dir}")
    n_chips = len(list(chips_dir.glob("*.tif")))
    if n_chips == 0:
        raise FileNotFoundError(f"no .tif chips found under {chips_dir}")
    if not labels_geojson.exists():
        raise FileNotFoundError(f"labels_geojson does not exist: {labels_geojson}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_chips < TINY_DATASET_WARN_THRESHOLD:
        log.warning(
            "fewshot buildings: only %d chips; the trainable surface is decoder + heads "
            "(millions of params) and will likely overfit. Consider more chips or fewer --epochs.",
            n_chips,
        )
    log.info(
        "fewshot buildings: %d chips, labels=%s, out=%s, epochs=%d, lr=%.1e",
        n_chips,
        labels_geojson.name,
        out_dir,
        epochs,
        lr,
    )
    cfg = _buildings_config()
    pretrained = _resolve_building_ckpt()
    summary = finetune(
        cfg=cfg,
        pretrained_ckpt=str(pretrained),
        chips_dir=str(chips_dir),
        labels_geojson=str(labels_geojson),
        out_dir=str(out_dir),
        val_frac=val_frac,
        ft_lr=lr,
        ft_epochs=epochs,
        ft_patience=patience,
    )
    delta = float(summary["delta"])
    if delta < 0:
        log.warning(
            "fewshot buildings: val IoU regressed by %.4f (pre=%.4f, post=%.4f). "
            "Overfit signal: keep the pretrained checkpoint instead.",
            delta,
            float(summary["val_iou_pretrained"]),
            float(summary["val_iou_finetuned"]),
        )
    return FewshotBuildingsResult(
        n_train=int(summary["n_train"]),
        n_val=int(summary["n_val"]),
        val_iou_pretrained=float(summary["val_iou_pretrained"]),
        val_iou_finetuned=float(summary["val_iou_finetuned"]),
        delta=delta,
        best_ckpt=Path(summary["best_ckpt"]),
        output_dir=Path(summary["output_dir"]),
    )


def fit_buildings_from_tm(
    project_ids: list[int],
    out_dir: Path,
    zoom: int = 19,
    imagery_tms_override: str | None = None,
    epochs: int = 10,
    lr: float = 5e-5,
    val_frac: float = 0.3,
    patience: int = 3,
    hpo_trials: int = HPO_DEFAULT_TRIALS,
    hpo_seeds: int = HPO_DEFAULT_SEEDS,
) -> FewshotBuildingsResult:
    """TM AOI + tile download + PostPass OSM labels, then fit (Optuna when hpo_trials>0)."""
    import asyncio

    from geomltoolkits.downloader.tms import download_tiles

    from dda.pipeline.postpass import fetch_postpass_buildings
    from dda.pipeline.tm import fetch_tm_project, union_aois

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles_root = out_dir / "tiles"
    tiles_root.mkdir(exist_ok=True)
    chips_dir = tiles_root / "chips"

    projects = [fetch_tm_project(pid) for pid in project_ids]
    if imagery_tms_override:
        log.warning("fewshot TM: forcing --imagery-tms %s (mapper's imagery ignored)", imagery_tms_override)
    used_projects: list = []
    for proj in projects:
        proj_tms = imagery_tms_override or proj.imagery_tms
        if not proj_tms:
            log.warning(
                "fewshot TM %d: no usable imagery URL, skipping (set imagery_tms_override to include)",
                proj.project_id,
            )
            continue
        aoi_path = out_dir / f"aoi_tm{proj.project_id}.geojson"
        aoi_path.write_text(json.dumps(proj.aoi))
        log.info("fewshot TM %d: z%d imagery=%s", proj.project_id, zoom, proj_tms)
        asyncio.run(
            download_tiles(
                tms=proj_tms,
                zoom=zoom,
                out=str(tiles_root),
                geojson=str(aoi_path),
                within=False,
                georeference=True,
                prefix=f"tm{proj.project_id}",
                extension="tif",
            )
        )
        _stretch_chip_dir(chips_dir, prefix=f"tm{proj.project_id}-")
        used_projects.append(proj)
    if not used_projects:
        raise RuntimeError(
            f"None of {project_ids} had usable imagery and no override was set; nothing to train on."
        )
    aoi_union = union_aois(used_projects)
    aoi_union_path = out_dir / "aoi_union.geojson"
    aoi_union_path.write_text(json.dumps(aoi_union))
    labels_path = out_dir / "labels.geojson"
    fetch_postpass_buildings(aoi_union_path, labels_path)
    if hpo_trials > 0:
        hpo = hpo_buildings(
            chips_dir=chips_dir,
            labels_geojson=labels_path,
            out_dir=out_dir,
            epochs=epochs,
            val_frac=val_frac,
            patience=patience,
            n_trials=hpo_trials,
            n_seeds=hpo_seeds,
        )
        return hpo.result
    return fit_buildings(
        chips_dir=chips_dir,
        labels_geojson=labels_path,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        val_frac=val_frac,
        patience=patience,
    )


@dataclass(frozen=True)
class HpoResult:
    """Return value of `hpo_buildings`: the best-fit model + the search trace."""

    result: FewshotBuildingsResult
    best_lr: float
    best_weight_decay: float
    best_trial_mean_iou: float
    best_trial_iou_std: float
    pretrained_iou: float
    use_pretrained_instead: bool
    trials: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.result.to_dict(),
            "best_lr": self.best_lr,
            "best_weight_decay": self.best_weight_decay,
            "best_trial_mean_iou": self.best_trial_mean_iou,
            "best_trial_iou_std": self.best_trial_iou_std,
            "pretrained_iou": self.pretrained_iou,
            "use_pretrained_instead": self.use_pretrained_instead,
            "trials": self.trials,
        }


def hpo_buildings(
    chips_dir: Path,
    labels_geojson: Path,
    out_dir: Path,
    epochs: int = 10,
    val_frac: float = 0.3,
    patience: int = 3,
    n_trials: int = HPO_DEFAULT_TRIALS,
    n_seeds: int = HPO_DEFAULT_SEEDS,
) -> HpoResult:
    """Optuna (lr, weight_decay) search scored by mean val_iou_finetuned across n_seeds splits."""
    import statistics

    import optuna
    from dinov3_hot.finetune import finetune

    from dda.pipeline.buildings import _resolve_building_ckpt

    chips_dir = Path(chips_dir).resolve()
    labels_geojson = Path(labels_geojson).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = out_dir / "hpo_trials"
    pretrained = _resolve_building_ckpt()

    trials_log: list[dict[str, Any]] = []
    pretrained_iou_ref: list[float] = []

    def _objective(trial: "optuna.trial.Trial") -> float:
        lr = trial.suggest_float("lr", *HPO_LR_RANGE, log=True)
        wd = trial.suggest_float("weight_decay", *HPO_WD_RANGE, log=True)
        seed_ious: list[float] = []
        pre_ious: list[float] = []
        for seed_i in range(n_seeds):
            cfg = _buildings_config()
            cfg.seed = 42 + seed_i
            trial_out = trials_dir / f"t{trial.number:02d}_s{seed_i}"
            summary = finetune(
                cfg=cfg,
                pretrained_ckpt=str(pretrained),
                chips_dir=str(chips_dir),
                labels_geojson=str(labels_geojson),
                out_dir=str(trial_out),
                val_frac=val_frac,
                ft_lr=lr,
                ft_weight_decay=wd,
                ft_epochs=epochs,
                ft_patience=patience,
            )
            seed_ious.append(float(summary["val_iou_finetuned"]))
            pre_ious.append(float(summary["val_iou_pretrained"]))
        mean_iou = statistics.mean(seed_ious)
        std_iou = statistics.stdev(seed_ious) if len(seed_ious) > 1 else 0.0
        trials_log.append(
            {
                "trial": trial.number,
                "lr": lr,
                "weight_decay": wd,
                "seed_ious": seed_ious,
                "mean_iou": mean_iou,
                "std_iou": std_iou,
                "pretrained_iou_mean": statistics.mean(pre_ious),
            }
        )
        pretrained_iou_ref.extend(pre_ious)
        log.info(
            "hpo trial %d: lr=%.2e wd=%.2e mean_iou=%.4f std=%.4f",
            trial.number,
            lr,
            wd,
            mean_iou,
            std_iou,
        )
        return mean_iou

    log.info(
        "hpo_buildings: %d trials x %d seeds = %d fits (lr %s, wd %s)",
        n_trials,
        n_seeds,
        n_trials * n_seeds,
        HPO_LR_RANGE,
        HPO_WD_RANGE,
    )
    study_db = out_dir / "hpo_study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        storage=f"sqlite:///{study_db}",
        study_name="fewshot_buildings",
        load_if_exists=True,
    )
    done = len(study.trials)
    if done > 0:
        log.info("hpo_buildings: resuming from %d completed trials", done)
    remaining = max(0, n_trials - done)
    if remaining > 0:
        study.optimize(_objective, n_trials=remaining, show_progress_bar=False)

    best_lr = float(study.best_params["lr"])
    best_wd = float(study.best_params["weight_decay"])
    best_mean = float(study.best_value)
    best_std = next(t["std_iou"] for t in trials_log if t["trial"] == study.best_trial.number)
    pretrained_iou = statistics.mean(pretrained_iou_ref) if pretrained_iou_ref else 0.0
    use_pretrained_instead = best_mean < pretrained_iou
    if use_pretrained_instead:
        log.warning(
            "hpo_buildings: best trial mean IoU %.4f is BELOW pretrained baseline %.4f. "
            "Ship the pretrained checkpoint; do NOT use the fine-tuned one.",
            best_mean,
            pretrained_iou,
        )
    log.info(
        "hpo_buildings: best lr=%.2e wd=%.2e mean_iou=%.4f (pretrained=%.4f)",
        best_lr,
        best_wd,
        best_mean,
        pretrained_iou,
    )

    final = fit_buildings_with_config(
        chips_dir=chips_dir,
        labels_geojson=labels_geojson,
        out_dir=out_dir,
        epochs=epochs,
        lr=best_lr,
        weight_decay=best_wd,
        val_frac=val_frac,
        patience=patience,
    )
    (out_dir / "hpo_report.json").write_text(
        json.dumps(
            {
                "best_lr": best_lr,
                "best_weight_decay": best_wd,
                "best_trial_mean_iou": best_mean,
                "best_trial_iou_std": best_std,
                "pretrained_iou_mean": pretrained_iou,
                "use_pretrained_instead": use_pretrained_instead,
                "n_trials": n_trials,
                "n_seeds_per_trial": n_seeds,
                "trials": trials_log,
            },
            indent=2,
        )
    )
    return HpoResult(
        result=final,
        best_lr=best_lr,
        best_weight_decay=best_wd,
        best_trial_mean_iou=best_mean,
        best_trial_iou_std=best_std,
        pretrained_iou=pretrained_iou,
        use_pretrained_instead=use_pretrained_instead,
        trials=trials_log,
    )


def fit_buildings_with_config(
    chips_dir: Path,
    labels_geojson: Path,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_frac: float = 0.3,
    patience: int = 3,
) -> FewshotBuildingsResult:
    """Same as `fit_buildings` but also overrides `weight_decay`. Used as the HPO final fit."""
    from dinov3_hot.finetune import finetune

    from dda.pipeline.buildings import _resolve_building_ckpt

    pretrained = _resolve_building_ckpt()
    cfg = _buildings_config()
    summary = finetune(
        cfg=cfg,
        pretrained_ckpt=str(pretrained),
        chips_dir=str(chips_dir),
        labels_geojson=str(labels_geojson),
        out_dir=str(out_dir),
        val_frac=val_frac,
        ft_lr=lr,
        ft_weight_decay=weight_decay,
        ft_epochs=epochs,
        ft_patience=patience,
    )
    delta = float(summary["delta"])
    return FewshotBuildingsResult(
        n_train=int(summary["n_train"]),
        n_val=int(summary["n_val"]),
        val_iou_pretrained=float(summary["val_iou_pretrained"]),
        val_iou_finetuned=float(summary["val_iou_finetuned"]),
        delta=delta,
        best_ckpt=Path(summary["best_ckpt"]),
        output_dir=Path(summary["output_dir"]),
    )
