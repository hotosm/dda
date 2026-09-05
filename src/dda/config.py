from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]

# xBD Joint Damage Scale; background is handled by the localization head, not a damage class.
DAMAGE_CLASSES = ("no-damage", "minor-damage", "major-damage", "destroyed")
N_DAMAGE_CLASSES = len(DAMAGE_CLASSES)


@dataclass
class TrainConfig:
    backbone: str = "terratorch_dinov3_vitl16"
    hf_ckpt_repo: str = "kshitijrajsharma/dinov3"
    hf_ckpt_file: str = "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
    # Trained damage checkpoint, auto-downloaded when no local --ckpt is given.
    model_repo: str = "kshitijrajsharma/dinov3-damage-assessment"
    model_file: str = "model.ckpt"

    img_size: int = 512
    patch_size: int = 16
    seg_out_indices: tuple[int, ...] = (5, 11, 17, 23)
    embed_dim: int = 1024
    decoder_channels: int = 256
    unfreeze_last_n: int = 0

    use_pre: bool = True
    modality_dropout: float = 0.3
    # Photometric jitter on the RGB tiles; forces event-invariant features for cross-event robustness.
    augment_photometric: bool = False

    dataset_repo: str = "EVER-Z/torchange_xView2"
    dataset_splits: tuple[str, ...] = ("train", "tier3")
    data_root: str = "data/xbd"
    data_pct: float = 100.0
    # Events held out of train so val/test measure cross-event generalisation, not memorisation.
    val_events: tuple[str, ...] = ("mexico-earthquake",)
    test_events: tuple[str, ...] = ("palu-tsunami", "joplin-tornado")
    # When val_events is empty, fall back to an in-distribution random split of this fraction.
    val_frac: float = 0.1

    batch_size: int = 4
    eval_batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    lr: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 50
    early_stop_patience: int = 6
    loc_loss_weight: float = 0.3
    dmg_loss_weight: float = 1.0
    ordinal_distance_power: float = 1.0
    class_weights: tuple[float, ...] | None = None
    onecycle_pct_start: float = 0.05
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1e4
    precision: str = "bf16-mixed"
    seed: int = 42
    grad_clip: float = 1.0

    output_dir: str = "outputs"
    run_name: str = "dda_v1"

    tile_window: int = 512
    tile_stride: int = 384
    # Per-scene percentile stretch of pre/post before inference; disaster scenes arrive uncalibrated.
    radiometric_normalize: bool = True
    pool_op: str = "percentile"
    pool_percentile: float = 80.0
    temperature: float = 1.0

    # OSM building source used when buildings.source == "osm" (portable across events).
    osm_source: str = "raw_data_api"
    osm_tag_families: list[dict[str, str]] = field(
        default_factory=lambda: [{"key": "building", "status": "standing"}]
    )

    # Damage output schema + provenance strings written into damage.geojson.
    damage_output_schema: list[str] = field(
        default_factory=lambda: [
            "osm_id",
            "osm_type",
            "building",
            "osm_status",
            "damage_class",
            "damage",
            "damage_confidence",
            "damage_model",
            "imagery_pre",
            "imagery_post",
        ]
    )
    damage_label_map: dict[int, str] = field(
        default_factory=lambda: {
            -1: "no-data",
            0: "no-damage",
            1: "minor-damage",
            2: "major-damage",
            3: "destroyed",
        }
    )
    damage_provenance_imagery_pre: str = ""
    damage_provenance_imagery_post: str = ""
    damage_provenance_damage_model: str = ""


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> DictConfig:
    base = OmegaConf.structured(TrainConfig)
    if path is not None:
        base = OmegaConf.merge(base, OmegaConf.load(Path(path)))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(overrides))
    return cast(DictConfig, base)


def resolve_root(cfg: DictConfig) -> Path:
    p = Path(cfg.data_root)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_output(cfg: DictConfig) -> Path:
    p = Path(cfg.output_dir)
    out = (p if p.is_absolute() else REPO_ROOT / p) / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    return out
