"""Publish a trained damage model (ONNX + checkpoint + config + calibration + card) to the Hub.

Usage:
    uv run python scripts/publish_model.py --repo-id <user>/<model> --ckpt <ckpt> --onnx <onnx> \
        --config <config.yaml> --temperature <T> --metrics-json <metrics.json>
"""

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _model_card(repo_id: str, temperature: float, metrics: dict) -> str:
    per_class = metrics.get("per_class_f1", {})
    rows = "\n".join(f"| {name} | {f} |" for name, f in per_class.items())
    split = metrics.get("split", "val")
    n_buildings = metrics.get("n_buildings", "?")
    return f"""---
license: cc-by-nc-sa-4.0
tags:
  - disaster-response
  - damage-assessment
  - remote-sensing
  - dinov3
  - building-damage
---

# {repo_id}

Building-level disaster damage assessment. A frozen DINOv3 ViT-L/16 satellite backbone with a
trainable UperNet decoder and two heads (localization + ordinal 4-class damage). Given building
footprints and post-disaster imagery (optionally pre-disaster imagery), it assigns each building a
damage level on the xBD Joint Damage Scale with a calibrated confidence.

## Damage classes

`no-damage`, `minor-damage`, `major-damage`, `destroyed`.

## Object-level metrics ({split} split, {n_buildings} buildings)

| class | F1 |
|---|---|
{rows}

Macro F1 {metrics.get("macro_f1", "?")} | harmonic damage F1 {metrics.get("harmonic_f1", "?")}.

The damage F1 is computed on building pixels only (background excluded). Numbers are
in-distribution (the xView2 benchmark splits by tile, so the same events appear in train and
test); cross-event generalisation to a fully unseen disaster is harder for the subtle
minor/major classes.

## Inputs and outputs

- Input: building footprints (GeoJSON) + post-disaster RGB GeoTIFF, optionally a pre-disaster
  GeoTIFF aligned to the post grid. Footprints must overlay the post image correctly.
- Output: the footprints annotated with `damage_class` (0..3, or -1 when the model could not
  assign a class), `damage` (class label), and `damage_confidence` (mean model probability of the
  assigned class over the footprint; NaN for no-data rows).

## Files

- `model.onnx`: self-contained inference graph (post, pre -> damage logits).
- `model.ckpt`: Lightning checkpoint for evaluation or further training.
- `config.yaml`: training configuration.
- `calibration.json`: confidence temperature ({temperature}).

## Confidence

Raw logits from the ONNX graph can be turned into calibrated probabilities with
`softmax(logits / {temperature})`. The pipeline's per-building `damage_confidence` is the mean
of the assigned class's probability over the footprint pixels.

## Backbone

DINOv3 ViT-L/16 (`sat493m`), frozen. Decoder, fusion, and heads are the only trained parameters
(~24M). Trained on xBD (xView2), license CC BY-NC-SA 4.0; this model inherits the non-commercial
share-alike terms.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the damage model to the HF Hub")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--metrics-json", type=Path)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    metrics = json.loads(args.metrics_json.read_text()) if args.metrics_json else {}
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_file(path_or_fileobj=str(args.ckpt), path_in_repo="model.ckpt", repo_id=args.repo_id)
    api.upload_file(path_or_fileobj=str(args.config), path_in_repo="config.yaml", repo_id=args.repo_id)
    if args.onnx and args.onnx.exists():
        api.upload_file(path_or_fileobj=str(args.onnx), path_in_repo="model.onnx", repo_id=args.repo_id)

    calibration = {"temperature": args.temperature, "metrics": metrics}
    api.upload_file(
        path_or_fileobj=json.dumps(calibration, indent=2).encode(),
        path_in_repo="calibration.json",
        repo_id=args.repo_id,
    )
    api.upload_file(
        path_or_fileobj=_model_card(args.repo_id, args.temperature, metrics).encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )
    log.info("Published to https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
