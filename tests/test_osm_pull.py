import json
import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import box

from dda.pipeline import osm_pull


def test_pull_fails_and_drops_the_cache_when_a_family_has_no_osm_id(tmp_path, monkeypatch):
    (tmp_path / "aoi.geojson").write_text("{}")
    monkeypatch.setattr(osm_pull, "_fetch_snapshot", lambda aoi, key, zip_out: zip_out.write_bytes(b"zip"))
    monkeypatch.setattr(
        osm_pull, "_extract_geojson", lambda zip_path: gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs=4326)
    )

    with pytest.raises(RuntimeError, match="no osm_id"):
        osm_pull.pull_osm_buildings(
            tmp_path / "aoi.geojson", tmp_path / "cache", [{"key": "building", "status": "standing"}]
        )

    assert not (tmp_path / "cache" / "rda_building.zip").exists()


BOUNDARY = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}


@pytest.mark.parametrize("export_first", [False, True])
def test_extract_reads_the_export_and_skips_the_clipping_boundary(tmp_path, export_first):
    building = {
        "type": "Feature",
        "properties": {"osm_id": 7},
        "geometry": {"type": "Point", "coordinates": [0, 0]},
    }
    members = [
        ("clipping_boundary.geojson", json.dumps(BOUNDARY)),
        ("RawExport.geojson", json.dumps({"type": "FeatureCollection", "features": [building]})),
    ]
    zip_path = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, content in reversed(members) if export_first else members:
            archive.writestr(name, content)

    assert osm_pull._extract_geojson(zip_path)["osm_id"].tolist() == [7]


def test_extract_fails_when_the_zip_holds_no_export(tmp_path):
    zip_path = tmp_path / "snapshot.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("clipping_boundary.geojson", json.dumps(BOUNDARY))

    with pytest.raises(RuntimeError, match="expected one export GeoJSON"):
        osm_pull._extract_geojson(zip_path)
