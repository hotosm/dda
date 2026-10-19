from pathlib import Path

from dda.pipeline import fetch


def test_fetch_raster_reads_a_tif_url_template_as_tiles(tmp_path, monkeypatch):
    calls = []

    def fake_tms(url, aoi, out_path, zoom):
        calls.append(url)
        Path(out_path).write_bytes(b"tiles")

    monkeypatch.setattr(fetch, "_fetch_tms", fake_tms)
    template = "https://tiles.example/{z}/{x}/{y}.tif"

    fetch.fetch_raster(template, tmp_path / "aoi.geojson", tmp_path / "pre.tif", zoom=18)

    assert calls == [template]
