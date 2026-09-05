## dda

Building-level disaster damage assessment with a frozen DINOv3 siamese encoder and a trainable
decoder.

The model takes building footprints and post-disaster satellite imagery (optionally pre-disaster
imagery as well) and assigns each building a damage level on the xBD Joint Damage Scale:
`no-damage`, `minor-damage`, `major-damage`, `destroyed`. Each prediction carries a calibrated
confidence so low-certainty buildings can be flagged for human review.

### Approach

- **Backbone**: DINOv3 ViT-L/16, frozen, loaded through TerraTorch. The default checkpoint is the
  satellite variant (`sat493m`, Maxar 0.6 m).
- **Siamese pre/post**: pre and post tiles share the backbone. Features are fused per tap and the
  model is trained with modality dropout, so one checkpoint serves both the pre+post and the
  post-only regime. Late feature fusion tolerates the misregistration that comes from footprints
  drawn on different imagery than the post-disaster image.
- **Decoder and heads**: a TerraTorch UperNet decoder feeds a binary localization head and a
  4-class damage head. The damage loss is class-weighted cross-entropy plus a squared Earth-Mover
  term, which penalises predictions by their ordinal distance from the truth.
- **Per-building output**: the dense damage map is pooled inside each input footprint to produce
  one class and one confidence per building.

### Results

Per-building damage F1 on the xBD (xView2) held-out test events (palu-tsunami, joplin-tornado;
6,415 buildings):

| class | F1 |
| --- | --- |
| no-damage | 0.92 |
| minor-damage | 0.61 |
| major-damage | 0.49 |
| destroyed | 0.91 |
| macro | 0.73 |
| harmonic (xView2 damage F1) | 0.68 |

Splits are held out by disaster event, so these measure cross-event generalisation.

Under the official xView2 pixel metric (`0.3 * localization F1 + 0.7 * damage F1`) on the public
xBD test split (933 images): localization 0.82, damage 0.73, overall **0.76**.

### Assessing a new area

See [`docs/new_area_assessment.md`](docs/new_area_assessment.md) for the full walkthrough
(AOI, imagery, config, optional label-in-region, optional fine-tuning).

Short version, one YAML declares the event and one command runs it:

```bash
dda run --config conf/colombia_eq.yaml
```

Stage order: `aoi, prepare, fewshot, buildings, damage, publish`. Resume from
a stage with `-s <stage>` or run one stage in isolation with `--only <stage>`. Override
any field from the CLI with OmegaConf dotlist syntax (`buildings.fewshot.hpo_trials=4`).
Outputs land under `outputs/<area>/`; the final deliverable is `outputs/<area>/damage.geojson`.
Labeling is out of scope for the pipeline; add a `damage` column to `buildings.geojson`
externally, then rerun `damage` (see `docs/new_area_assessment.md` section 6).

The per-stage commands (`dda prepare`, `dda buildings`, `dda damage`, `dda publish`) are
available for one-off experiments and mid-pipeline resumes.

### CLI

Per-area pipeline:

| command | purpose |
| --- | --- |
| `dda run` | run the whole pipeline from one event YAML |
| `dda prepare` | fetch pre + post rasters, coregister pre onto post, per-scene stretch |
| `dda buildings` | detect footprints (`fair`), pull OSM (`--source osm`), or use your own (`--input`) |
| `dda damage` | score each building, write `damage.geojson` |
| `dda eval` | per-class F1 and confusion matrix vs a labelled ground-truth GeoJSON |
| `dda publish` | upload the area folder to a Hugging Face dataset |

Model adaptation:

| command | purpose |
| --- | --- |
| `dda fewshot buildings` | backbone-frozen fine-tune of the fAIr buildings model on a TM project or local chips + labels |
| `dda fewshot damage` | fine-tune the damage model on one area (strict head-only by default) |

Training and dev-time (`dda train`, `dda predict`, `dda export`, `dda evaluate`, `dda calibrate`)
are documented in the per-command `--help`.

### Data

Training reads xBD (xView2) from `kshitijrajsharma/xview2-xbd` on the Hugging Face Hub.
Splits are held out by disaster event. The published dataset covers training and evaluation
directly. To rebuild a copy from a non-gated mirror, `scripts/build_xbd_dataset.py` assembles
the pre/post tiles, damage rasters, and polygons.

### Docker

```
docker build -t dda-gpu:local .
docker run --rm --gpus all \
  -v $PWD/outputs:/data/outputs \
  -v $HOME/.cache/huggingface:/data/hf-cache \
  -v $HOME/.cache/torch:/data/cache/torch \
  -v $PWD/conf:/data/conf:ro \
  dda-gpu:local run --config /data/conf/colombia_eq.yaml --outputs-root /data/outputs
```

The base is the Astral `uv` slim Python 3.13 image; torch bundles its own CUDA runtime. The
`facebookresearch/dinov3` torch.hub source is baked in, so the backbone loads offline at
runtime. Volumes carry the outputs directory and the HF, torch caches so downloaded
checkpoints persist between runs.

For in-region fine-tuning, label a copy of `outputs/<area>/buildings.geojson` in whatever
GeoJSON-aware tool you prefer by adding a `damage` column with values in
`{no-damage, minor-damage, major-damage, destroyed}` (or integer codes `1..4`), then feed
it to `dda fewshot damage --labels <that.geojson>`.

### Stack

`uv`, PyTorch Lightning, TerraTorch, OmegaConf, rasterio, geopandas, sklearn. Lint and
type-check with `ruff` and `ty` via `just lint`.

Both the damage and buildings models use a frozen DINOv3 ViT-L/16 backbone. The damage model
loads the satellite-pretrained variant (`sat493m`); the buildings model loads the
web-pretrained variant (`lvd1689m`). Cached in `~/.cache/huggingface/` after first use.
