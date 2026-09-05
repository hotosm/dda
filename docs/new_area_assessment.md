## Assessing a new area

End-to-end recipe for running `dda` over a fresh AOI after a disaster. The example throughout
is the August 2026 Colombia earthquake around Pereira; substitute your own area name, AOI,
and imagery URLs.

Two ways to run it:

- **One YAML, one command**: write an event config, run `dda run --config <file>`. Recommended
  for full assessments and for reproducibility.
- **Per-stage commands**: `dda prepare`, `dda buildings`, `dda damage`, `dda publish`.
  Useful for one-off experiments and for resuming mid-pipeline.

### 1. Get the AOI

An AOI is a single GeoJSON polygon (or MultiPolygon) in `EPSG:4326`. Keep it tight since the
pipeline processes everything inside the bounding box.

If a HOT Tasking Manager project already covers the area, either point the config at the
project ID via `tm_aoi_project`, or pull it down manually:

```bash
curl -s https://tasking-manager-production-api.hotosm.org/api/v2/projects/<project_id>/ \
  | jq '.areaOfInterest' > aoi.geojson
```

Otherwise draw the polygon in QGIS or [geojson.io](https://geojson.io) and save as
`aoi_columbia.geojson`.

### 2. Pick pre and post imagery

Two sources per raster: a **TMS URL** with `{z}/{x}/{y}` placeholders, or a **Cloud Optimized
GeoTIFF (COG)** URL. Either works for `pre_img` and `post_img`.

Common HOT sources:

| source | when to use | shape |
| --- | --- | --- |
| OpenAerialMap | HOT, Vantor, Planet open-data VHR for the event | COG URL from `api.openaerialmap.org/meta?bbox=...` |
| ESRI World Imagery | fallback pre-imagery when no VHR pre exists | `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}` |
| GSI seamlessphoto (Japan) | Japan-specific pre-imagery | `https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg` |
| Bing | permissive commercial-license pre-imagery | pass hostname; see `--pre-img` help |

Pick the same-sensor pair when possible. When same-sensor pre is not available, use ESRI as
the pre; the pipeline coregisters and photometric-calibrates it against the post grid.

### 3. Write the event config

One YAML declares the whole event. Colombia's config lives at `conf/colombia_eq.yaml`:

```yaml
area: colombia_eq_pereira
outputs_root: outputs

aoi: aoi_columbia.geojson
tm_aoi_project: null

pre_img:  https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/6a7cb2bef7b084f6608726ad/0/6a7cb2bef7b084f6608726ae.tif
post_img: https://oin-hotosm-temp.s3.us-east-1.amazonaws.com/6a7cb280f7b084f660872669/0/6a7cb280f7b084f66087266a.tif
zoom: 19                           # ignored when source is a COG
photometric_calibration: true
stretch_percentiles: true
keep_raw: false                    # true to keep raw pre.tif + post.tif for QGIS inspection

buildings:
  source: fair                     # fair | osm | input
  ckpt: null                       # fine-tuned ckpt override; auto-set by fewshot if run
  fewshot:
    tm_projects: [61749]           # San Jose del Palmar mapping (region prior)
    hpo_trials: 8
    hpo_seeds: 1
    epochs: 10

damage:
  ckpt: null                       # override the auto-resolved HF damage ckpt

publish:
  enabled: false
  repo_id: null                    # e.g. hotosm/colombia_eq_2026
```

See `conf/event.example.yaml` for the annotated template.

### 4. Run the pipeline

Full end-to-end:

```bash
dda run --config conf/colombia_eq.yaml
```

Stage order: `aoi, prepare, fewshot, buildings, damage, publish`. Labeling itself lives outside the pipeline; see section 6.

To resume from a specific stage through the end (earlier stages assumed done):

```bash
dda run --config conf/colombia_eq.yaml -s damage
```

To run one stage in isolation (useful when stepping through the pipeline for the first time):

```bash
dda run --config conf/colombia_eq.yaml --only prepare
```

Override any field from the CLI without touching the YAML (OmegaConf dotlist syntax):

```bash
dda run --config conf/colombia_eq.yaml buildings.fewshot.hpo_trials=4
dda run --config conf/colombia_eq.yaml damage.ckpt=outputs/colombia_eq_pereira/fs_damage/best.ckpt
```

Preview the plan without running:

```bash
dda run --config conf/colombia_eq.yaml --dry-run
```

### 5. Options during the run

**Buildings source.** Default `source: fair` runs the fAIr detector on the aligned pre. Two
alternatives:

```yaml
buildings:
  source: osm                      # fetch footprints from PostPass
```

```yaml
buildings:
  source: input
  input: my_footprints.geojson     # your own polygons
```

**Photometric calibration.** On by default. Set `photometric_calibration: false` when pre and
post already share a sensor and radiometry matches.

**Keep raw imagery.** `keep_raw: true` preserves the raw `pre.tif` and `post.tif` alongside
the aligned versions for QGIS inspection; default deletes them once the aligned files land.

### 6. Label in region (external)

Labeling is deliberately out of scope for this repo. The output of `dda buildings`,
`outputs/<area>/buildings.geojson`, is a plain GeoJSON of building polygons and can be
labeled in any tool that reads and writes GeoJSON.

Add a `damage` column with values in `{no-damage, minor-damage, major-damage, destroyed}`
(or integer codes `1..4`). Anything you leave unlabeled is ignored by the fine-tune.
Un-classified buildings, if you want to be explicit, can carry code `5` or
`"un-classified"`, which the loss also ignores.

### 7. Fine-tune in region

Head-only fit on a small labeled set, either buildings or damage.

Buildings from a Tasking Manager project (auto-fetches AOI, imagery, and OSM labels; this is
what the `fewshot` stage runs when the config lists `buildings.fewshot.tm_projects`):

```bash
dda fewshot buildings \
  --tm-projects 61749 \
  --out-dir outputs/colombia_eq_pereira/fs_buildings
```

Damage from your labels GeoJSON:

```bash
dda fewshot damage \
  --pre outputs/colombia_eq_pereira/pre_aligned.tif \
  --post outputs/colombia_eq_pereira/post_aligned.tif \
  --labels outputs/colombia_eq_pereira/damage_labels.geojson \
  --out-dir outputs/colombia_eq_pereira/fs_damage
```

Default is strict head-only; add `--full-decoder` when the label set is large enough to
justify unfreezing the pyramid and UperNet decoder.

Point the event YAML at the tuned checkpoint and rerun the damage stage:

```yaml
damage:
  ckpt: outputs/colombia_eq_pereira/fs_damage/best.ckpt
```

```bash
dda run --config conf/colombia_eq.yaml --only damage
```

### 8. Verify

If ground-truth labels are available, score the deliverable:

```bash
dda eval \
  --predictions outputs/colombia_eq_pereira/damage.geojson \
  --labels ground_truth.geojson \
  --out outputs/colombia_eq_pereira/metrics.json
```

Prints per-class precision, recall, F1, and the confusion matrix.

### What each stage writes

| file | stage | notes |
| --- | --- | --- |
| `outputs/<area>/aoi.geojson` | aoi | copied from `aoi:` or fetched from `tm_aoi_project:` |
| `outputs/<area>/pre.tif`, `post.tif` | prepare | deleted after coregistration unless `keep_raw: true` |
| `outputs/<area>/pre_aligned.tif` | prepare | pre reprojected onto post grid, drift-corrected, photometric-calibrated, per-scene stretched |
| `outputs/<area>/post_aligned.tif` | prepare | post copy with the same per-scene stretch applied |
| `outputs/<area>/coreg/drift.json`, `checkerboard.png` | prepare | drift record and visual alignment check |
| `outputs/<area>/fs_buildings/` | fewshot | HPO trials, best checkpoint, training log |
| `outputs/<area>/buildings.geojson` | buildings | building footprints |
| `outputs/<area>/damage.geojson` | damage | final deliverable, one class + confidence per building |

Every building in `damage.geojson` carries: `damage` (class label), `damage_class` (0..3, or
-1 when the model could not assign a class), and `damage_confidence` (mean model probability of
the assigned class over the footprint; NaN for no-data rows).

### Docker

The full pipeline runs inside the packaged image. Mount volumes for the outputs directory
and the HF, torch caches:

```bash
docker run --rm --gpus all \
  -v $PWD/outputs:/data/outputs \
  -v $HOME/.cache/huggingface:/data/hf-cache \
  -v $HOME/.cache/torch:/data/cache/torch \
  -v $PWD/conf:/data/conf:ro \
  -v $PWD/aoi_columbia.geojson:/data/aoi_columbia.geojson:ro \
  dda-gpu:local run --config /data/conf/colombia_eq.yaml --outputs-root /data/outputs
```

Cache volumes avoid re-downloading model checkpoints on every container start.

Labeling itself lives outside this repo, see section 6.
