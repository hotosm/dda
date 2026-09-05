"""OmegaConf schema for `dda run --config event.yaml`; one YAML declares an event end-to-end."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import MISSING, DictConfig, OmegaConf

_VALID_OSM_SOURCES = ("raw_data_api", "postpass")


@dataclass
class FewshotBuildingsConfig:
    tm_projects: list[int] = field(default_factory=list)
    imagery_tms_override: str | None = None
    hpo_trials: int = 8
    hpo_seeds: int = 1
    epochs: int = 10
    val_frac: float = 0.3
    patience: int = 3
    lr: float = 5e-5
    zoom: int = 19


@dataclass
class BuildingsConfig:
    source: str = "fair"
    input: str | None = None
    ckpt: str | None = None
    fewshot: FewshotBuildingsConfig = field(default_factory=FewshotBuildingsConfig)


@dataclass
class DamageConfig:
    ckpt: str | None = None


@dataclass
class PublishConfig:
    enabled: bool = False
    repo_id: str | None = None


@dataclass
class EventConfig:
    area: str = MISSING
    outputs_root: str = "outputs"
    aoi: str | None = None
    tm_aoi_project: int | None = None
    pre_img: str = MISSING
    post_img: str = MISSING
    zoom: int = 19
    photometric_calibration: bool = True
    stretch_percentiles: bool = True
    keep_raw: bool = False
    shift_direction: str = "pre_to_post"
    buildings: BuildingsConfig = field(default_factory=BuildingsConfig)
    damage: DamageConfig = field(default_factory=DamageConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    # OSM pull backend when buildings.source == "osm"; mirrors TrainConfig for provenance handoff.
    osm_source: str = "raw_data_api"
    osm_tag_families: list[dict[str, str]] = field(
        default_factory=lambda: [{"key": "building", "status": "standing"}]
    )

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


def load_event_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load YAML into the EventConfig schema; overrides use OmegaConf dotlist syntax."""
    base = OmegaConf.structured(EventConfig)
    loaded = OmegaConf.load(Path(path))
    merged = OmegaConf.merge(base, loaded)
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    if merged.osm_source not in _VALID_OSM_SOURCES:
        raise ValueError(f"osm_source must be one of {_VALID_OSM_SOURCES}, got {merged.osm_source!r}")
    return cast(DictConfig, merged)
