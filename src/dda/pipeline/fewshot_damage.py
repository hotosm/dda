"""Few-shot fine-tune of the damage model on one labelled area. `strict_head=True` (default)
freezes decoder + pyramid + fusion; only the two classifier heads train."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import rasterio
import torch
from huggingface_hub import hf_hub_download
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

from dda.config import load_config
from dda.data import DINOV3_MEAN, DINOV3_STD
from dda.infer import resolve_ckpt
from dda.losses import IGNORE_INDEX
from dda.model import DinoV3DamageLit
from dda.pipeline.hub_utils import enable_offline_torch_hub_fallback

log = logging.getLogger(__name__)

DAMAGE_LABEL_MAP: dict[str | int, int] = {
    "no-damage": 1,
    "no_damage": 1,
    "nodamage": 1,
    "minor-damage": 2,
    "minor_damage": 2,
    "minor": 2,
    "major-damage": 3,
    "major_damage": 3,
    "major": 3,
    "destroyed": 4,
    "un-classified": 5,
    "unclassified": 5,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}

TINY_DATASET_WARN_THRESHOLD = 20
MIN_PRE_COVERAGE_PCT = 90


@dataclass(frozen=True)
class FewshotDamageResult:
    n_train: int
    n_val: int
    pretrained_macro_f1: float
    finetuned_macro_f1: float
    delta: float
    best_ckpt: Path
    output_dir: Path
    strict_head: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "pretrained_macro_f1": self.pretrained_macro_f1,
            "finetuned_macro_f1": self.finetuned_macro_f1,
            "delta": self.delta,
            "best_ckpt": str(self.best_ckpt),
            "output_dir": str(self.output_dir),
            "strict_head": self.strict_head,
        }


def _normalize_damage_value(v: Any) -> int:
    """GeoPandas stringifies mixed-type columns, so numeric strings decode to int first."""
    if isinstance(v, str):
        s = v.strip().lower()
        try:
            key: str | int = int(s)
        except ValueError:
            key = s
    else:
        key = int(v)
    if key not in DAMAGE_LABEL_MAP:
        raise ValueError(
            f"labels: unrecognised damage value {v!r}. Expected one of "
            f"{sorted({k for k in DAMAGE_LABEL_MAP if not isinstance(k, int)})} "
            f"or integers 1..4 (xBD Joint Damage Scale)."
        )
    return DAMAGE_LABEL_MAP[key]


def _read_labels(labels_geojson: Path) -> list[tuple[Any, int]]:
    import geopandas as gpd

    gdf = gpd.read_file(labels_geojson)
    if "damage" not in gdf.columns:
        raise ValueError(
            f"labels {labels_geojson} must have a `damage` column with values in "
            f"{{1,2,3,4}} or string names (no-damage/minor-damage/major-damage/destroyed)."
        )
    return [(row.geometry, _normalize_damage_value(row.damage)) for row in gdf.itertuples()]


def _chip_area(
    pre_raster: Path,
    post_raster: Path,
    labels_geojson: Path,
    tile_size: int,
    stride: int,
) -> list[dict[str, Any]]:
    """Emit one dict per window that has a labelled building and enough pre coverage.

    Pre-coverage gate keeps siamese change from firing on empty pre tiles.
    """
    import geopandas as gpd

    labels = _read_labels(labels_geojson)
    with rasterio.open(post_raster) as post:
        height, width = post.height, post.width
        post_transform = post.transform
        post_crs = post.crs

        gdf = gpd.read_file(labels_geojson).to_crs(post_crs)
        shapes = [(g, code) for g, (_, code) in zip(gdf.geometry, labels, strict=True)]
        damage_raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=post_transform,
            fill=0,
            dtype="uint8",
        )
        log.info(
            "fewshot damage: rasterized %d labels into %dx%d damage raster (%s)",
            len(labels),
            height,
            width,
            {c: int((damage_raster == c).sum()) for c in [1, 2, 3, 4]},
        )

        chips: list[dict[str, Any]] = []
        with (
            rasterio.open(pre_raster) as pre_src,
            WarpedVRT(
                pre_src,
                crs=post_crs,
                transform=post_transform,
                width=width,
                height=height,
                resampling=Resampling.cubic,
            ) as pre_vrt,
        ):
            for y in range(0, height - tile_size + 1, stride):
                for x in range(0, width - tile_size + 1, stride):
                    dmg = damage_raster[y : y + tile_size, x : x + tile_size]
                    if int((dmg > 0).sum()) == 0:
                        continue
                    win = Window(x, y, tile_size, tile_size)  # ty: ignore[too-many-positional-arguments]
                    pre_tile = pre_vrt.read([1, 2, 3], window=win).astype(np.uint8)
                    if (pre_tile > 0).any(axis=0).mean() * 100 < MIN_PRE_COVERAGE_PCT:
                        continue
                    post_tile = post.read([1, 2, 3], window=win).astype(np.uint8)
                    chips.append({"post": post_tile, "pre": pre_tile, "damage": dmg.copy()})
    log.info("fewshot damage: kept %d chips (tile=%d, stride=%d)", len(chips), tile_size, stride)
    return chips


class _ChipDataset(Dataset):
    _MEAN = torch.tensor(DINOV3_MEAN, dtype=torch.float32).view(3, 1, 1)
    _STD = torch.tensor(DINOV3_STD, dtype=torch.float32).view(3, 1, 1)

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:  # ty: ignore[invalid-method-override]
        it = self.items[i]
        post_t = ((torch.from_numpy(it["post"]).float() / 255.0) - self._MEAN) / self._STD
        pre_t = ((torch.from_numpy(it["pre"]).float() / 255.0) - self._MEAN) / self._STD
        dmg_np = it["damage"].astype(np.int64)
        damage = torch.full(dmg_np.shape, IGNORE_INDEX, dtype=torch.long)
        building = (dmg_np >= 1) & (dmg_np <= 4)
        damage[torch.from_numpy(building)] = torch.from_numpy(dmg_np[building] - 1)
        build_mask = torch.from_numpy(building.astype(np.float32))
        return {"post": post_t, "pre": pre_t, "damage": damage, "build_mask": build_mask}


def _split_datasets(
    chips: list[dict[str, Any]], val_frac: float, seed: int
) -> tuple[_ChipDataset, _ChipDataset]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(chips))
    n_val = max(1, int(len(chips) * val_frac))
    val_items = [chips[i] for i in perm[:n_val]]
    train_items = [chips[i] for i in perm[n_val:]]
    return _ChipDataset(train_items), _ChipDataset(val_items)


def _apply_strict_head_freeze(model: DinoV3DamageLit) -> tuple[int, int]:
    """Freeze pyramid + decoder + fusion so only loc_head + dmg_head remain trainable."""
    net = model.net
    for module in (net.decoder, net.pyramid):
        for p in module.parameters():
            p.requires_grad = False
    if net.fusion is not None:
        for p in net.fusion.parameters():
            p.requires_grad = False
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in net.parameters() if not p.requires_grad)
    return n_trainable, n_frozen


def fit_damage(
    pre_raster: Path,
    post_raster: Path,
    labels_geojson: Path,
    out_dir: Path,
    ckpt: Path | None = None,
    epochs: int = 10,
    lr: float = 1e-4,
    tile_size: int = 512,
    stride: int = 384,
    val_frac: float = 0.3,
    patience: int = 3,
    batch_size: int = 2,
    strict_head: bool = True,
    seed: int = 42,
) -> FewshotDamageResult:
    """Fit the damage model to one area's labelled buildings.

    `strict_head=True` (default) is the real few-shot regime: only loc_head + dmg_head train.
    Labels GeoJSON must have a `damage` column with xBD integers 1..4 or string names.
    """
    pre_raster = Path(pre_raster).resolve()
    post_raster = Path(post_raster).resolve()
    labels_geojson = Path(labels_geojson).resolve()
    out_dir = Path(out_dir).resolve()
    for p in (pre_raster, post_raster, labels_geojson):
        if not p.exists():
            raise FileNotFoundError(str(p))
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(seed, workers=True)
    chips = _chip_area(pre_raster, post_raster, labels_geojson, tile_size, stride)
    if not chips:
        raise RuntimeError(
            "no valid chips: no window contained a labelled building with sufficient pre coverage"
        )
    if len(chips) < TINY_DATASET_WARN_THRESHOLD and not strict_head:
        log.warning(
            "fewshot damage: only %d chips and strict_head=False; the trainable decoder+heads "
            "surface will likely overfit. Pass strict_head=True (default) or supply more labels.",
            len(chips),
        )

    train_ds, val_ds = _split_datasets(chips, val_frac=val_frac, seed=seed)
    log.info("fewshot damage: split %d train / %d val", len(train_ds), len(val_ds))
    if len(train_ds) < 2 or len(val_ds) < 2:
        raise RuntimeError(
            f"fewshot damage needs at least 2 train + 2 val chips (BatchNorm collapses on "
            f"single-sample batches); got {len(train_ds)} train / {len(val_ds)} val. "
            "Reduce --tile-size / --stride to yield more chips, or supply more labels."
        )

    enable_offline_torch_hub_fallback()
    cfg = load_config(None)
    encoder_ckpt = hf_hub_download(repo_id=cfg.hf_ckpt_repo, filename=cfg.hf_ckpt_file)
    pretrained_ckpt = str(ckpt) if ckpt else resolve_ckpt(cfg, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoV3DamageLit.load_from_checkpoint(
        pretrained_ckpt, map_location=device, ckpt_path=encoder_ckpt, weights_only=False
    )
    model.lr = lr

    if strict_head:
        n_train_p, n_frozen_p = _apply_strict_head_freeze(model)
        log.info(
            "fewshot damage: strict_head=True; %.1fM trainable / %.1fM frozen params",
            n_train_p / 1e6,
            n_frozen_p / 1e6,
        )
    else:
        n_train_p = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
        log.info(
            "fewshot damage: strict_head=False; %.1fM trainable params (decoder + heads)",
            n_train_p / 1e6,
        )

    train_bs = min(batch_size, len(train_ds))
    val_bs = min(batch_size, len(val_ds))
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=0, drop_last=True)

    baseline_trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        inference_mode=False,
    )
    pre_metrics = baseline_trainer.validate(model, dataloaders=val_loader, verbose=False)[0]
    pre_macro_f1 = float(pre_metrics.get("val/dmg_macro_f1", 0.0))

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "ckpts",
        filename="ft-{epoch:02d}-{val/dmg_macro_f1:.4f}",
        monitor="val/dmg_macro_f1",
        mode="max",
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    early = EarlyStopping(monitor="val/dmg_macro_f1", mode="max", patience=patience)
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision="32" if not torch.cuda.is_available() else cfg.precision,
        accelerator="auto",
        devices=1,
        gradient_clip_val=cfg.grad_clip,
        callbacks=[ckpt_cb, early],
        logger=CSVLogger(save_dir=str(out_dir), name="lightning"),
        log_every_n_steps=1,
        default_root_dir=str(out_dir),
        inference_mode=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    post_macro_f1 = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    delta = post_macro_f1 - pre_macro_f1
    if delta < 0:
        log.warning(
            "fewshot damage: val macro-F1 regressed by %.4f (pre=%.4f, post=%.4f). "
            "Overfit signal: prefer the pretrained checkpoint over this fine-tuned one.",
            delta,
            pre_macro_f1,
            post_macro_f1,
        )
    return FewshotDamageResult(
        n_train=len(train_ds),
        n_val=len(val_ds),
        pretrained_macro_f1=pre_macro_f1,
        finetuned_macro_f1=post_macro_f1,
        delta=delta,
        best_ckpt=Path(ckpt_cb.best_model_path),
        output_dir=out_dir,
        strict_head=strict_head,
    )
