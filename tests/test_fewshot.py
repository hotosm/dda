"""Tests for `dda.pipeline.fewshot`.

Fast-suite tests: TM helper glue, precondition checks, mutually-exclusive-arg validation. The
actual `dinov3_hot.finetune` call is replaced by a stub, so no torch training runs here.
"""

import json
from pathlib import Path

import pytest

from dda.pipeline import fewshot, tm


def _fake_finetune_summary(out_dir: Path) -> dict:
    return {
        "n_train": 20,
        "n_val": 10,
        "val_iou_pretrained": 0.42,
        "val_iou_finetuned": 0.61,
        "delta": 0.19,
        "best_ckpt": str(out_dir / "ckpts" / "ft-05-0.6100.ckpt"),
        "output_dir": str(out_dir),
    }


def test_fit_buildings_missing_chips_dir(tmp_path):
    labels = tmp_path / "labels.geojson"
    labels.write_text("{}")
    with pytest.raises(FileNotFoundError, match="chips_dir does not exist"):
        fewshot.fit_buildings(tmp_path / "nope", labels, tmp_path / "out")


def test_fit_buildings_empty_chips_dir(tmp_path):
    chips = tmp_path / "chips"
    chips.mkdir()
    labels = tmp_path / "labels.geojson"
    labels.write_text("{}")
    with pytest.raises(FileNotFoundError, match=r"no \.tif chips"):
        fewshot.fit_buildings(chips, labels, tmp_path / "out")


def test_fit_buildings_missing_labels(tmp_path):
    chips = tmp_path / "chips"
    chips.mkdir()
    (chips / "a.tif").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="labels_geojson does not exist"):
        fewshot.fit_buildings(chips, tmp_path / "missing.geojson", tmp_path / "out")


def test_fit_buildings_delegates_to_dinov3_finetune(tmp_path, monkeypatch):
    chips = tmp_path / "chips"
    chips.mkdir()
    (chips / "a.tif").write_bytes(b"x")
    labels = tmp_path / "labels.geojson"
    labels.write_text('{"type":"FeatureCollection","features":[]}')

    captured = {}

    def _fake_finetune(**kwargs):
        captured.update(kwargs)
        return _fake_finetune_summary(Path(kwargs["out_dir"]))

    def _fake_resolve():
        return Path("/fake/model.ckpt")

    monkeypatch.setattr("dinov3_hot.finetune.finetune", _fake_finetune)
    monkeypatch.setattr("dda.pipeline.buildings._resolve_building_ckpt", _fake_resolve)
    monkeypatch.setattr(fewshot, "_buildings_config", object)

    out = tmp_path / "out"
    result = fewshot.fit_buildings(chips, labels, out, epochs=3, lr=1e-4)

    assert captured["ft_epochs"] == 3
    assert captured["ft_lr"] == 1e-4
    assert Path(captured["chips_dir"]) == chips.resolve()
    assert Path(captured["labels_geojson"]) == labels.resolve()
    assert Path(captured["pretrained_ckpt"]) == Path("/fake/model.ckpt")
    assert result.delta == pytest.approx(0.19)
    assert result.val_iou_finetuned == pytest.approx(0.61)


def test_fit_buildings_from_tm_all_null_imagery_needs_override(tmp_path, monkeypatch):
    aoi_fc = {"type": "FeatureCollection", "features": []}
    monkeypatch.setattr("dda.pipeline.tm.fetch_tm_project", lambda pid: tm.TMProject(pid, "x", aoi_fc, None))
    with pytest.raises(RuntimeError, match="no override was set"):
        fewshot.fit_buildings_from_tm([1, 2], tmp_path / "out")


def test_fit_buildings_from_tm_happy_path(tmp_path, monkeypatch):
    poly = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}
    aoi_fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": poly, "properties": {}}],
    }
    projects = {
        7: tm.TMProject(7, "seven", aoi_fc, "https://tiles/{z}/{x}/{y}.png"),
        8: tm.TMProject(8, "eight", aoi_fc, "https://tiles/{z}/{x}/{y}.png"),
    }

    def _fake_fetch(pid):
        return projects[pid]

    async def _fake_download_tiles(**kwargs):
        prefix = kwargs["prefix"]
        (Path(kwargs["out"]) / f"{prefix}-1-1.tif").write_bytes(b"x")
        return kwargs["out"]

    called = {}

    def _fake_postpass(aoi_path, out_path, timeout=180):
        called["aoi_path"] = aoi_path
        out_path.write_text('{"type":"FeatureCollection","features":[]}')
        return out_path

    def _fake_fit_buildings(chips_dir, labels_geojson, out_dir, **kwargs):
        called["chips_dir"] = chips_dir
        called["labels_geojson"] = labels_geojson
        called["kwargs"] = kwargs
        return fewshot.FewshotBuildingsResult(
            n_train=1,
            n_val=1,
            val_iou_pretrained=0.1,
            val_iou_finetuned=0.2,
            delta=0.1,
            best_ckpt=Path(out_dir) / "ckpt",
            output_dir=Path(out_dir),
        )

    monkeypatch.setattr("dda.pipeline.tm.fetch_tm_project", _fake_fetch)
    monkeypatch.setattr("geomltoolkits.downloader.tms.download_tiles", _fake_download_tiles)
    monkeypatch.setattr("dda.pipeline.postpass.fetch_postpass_buildings", _fake_postpass)
    monkeypatch.setattr(fewshot, "fit_buildings", _fake_fit_buildings)

    out_dir = tmp_path / "fs"
    result = fewshot.fit_buildings_from_tm([7, 8], out_dir, epochs=2, lr=1e-4, hpo_trials=0)

    assert result.delta == pytest.approx(0.1)
    assert (out_dir / "aoi_tm7.geojson").exists()
    assert (out_dir / "aoi_tm8.geojson").exists()
    assert (out_dir / "aoi_union.geojson").exists()
    union = json.loads((out_dir / "aoi_union.geojson").read_text())
    assert {f["properties"]["tm_project_id"] for f in union["features"]} == {7, 8}
    assert called["kwargs"]["epochs"] == 2
    assert called["kwargs"]["lr"] == 1e-4
