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
