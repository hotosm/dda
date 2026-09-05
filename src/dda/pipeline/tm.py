"""HOT Tasking Manager v2 client; `.imagery` is often null, so callers must handle None or override."""

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

TM_API = "https://tasking-manager-production-api.hotosm.org/api/v2"
USER_AGENT = "dda-pipeline/0.1 (contact: krschap@duck.com)"

_ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
_TM_IMAGERY_LABELS: dict[str, str] = {
    "esri": _ESRI_URL,
    "esri world imagery": _ESRI_URL,
    "bing": _ESRI_URL,
    "mapbox satellite": _ESRI_URL,
    "maxar-premium": _ESRI_URL,
    "maxar-standard": _ESRI_URL,
}


def _normalise_imagery(value: Any) -> str | None:
    """Return an XYZ URL; TM labels are mapped, unknown non-URL strings become None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not v:
        return None
    if v.lower().startswith(("http://", "https://")):
        return v
    return _TM_IMAGERY_LABELS.get(v.lower())


@dataclass(frozen=True)
class TMProject:
    project_id: int
    name: str
    aoi: dict[str, Any]
    imagery_tms: str | None


def fetch_tm_project(project_id: int, timeout: int = 60) -> TMProject:
    url = f"{TM_API}/projects/{project_id}/"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"TM project {project_id} HTTP {err.code}: {err.reason}") from err

    aoi = payload.get("areaOfInterest")
    if not aoi or not isinstance(aoi, dict):
        raise RuntimeError(
            f"TM project {project_id}: `.areaOfInterest` missing or malformed "
            f"(got {type(aoi).__name__}); cannot proceed."
        )
    if aoi.get("type") in {"Polygon", "MultiPolygon"}:
        aoi = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": aoi, "properties": {}}],
        }
    imagery = _normalise_imagery(payload.get("imagery"))

    info = payload.get("projectInfo") or {}
    name = info.get("name") or f"tm-{project_id}"
    log.info(
        "TM project %d: name=%r, imagery=%s, AOI features=%d",
        project_id,
        name,
        "<set>" if imagery else "<null>",
        len(aoi.get("features", [])) if aoi.get("type") == "FeatureCollection" else 1,
    )
    return TMProject(project_id=project_id, name=name, aoi=aoi, imagery_tms=imagery)


def union_aois(projects: list[TMProject]) -> dict[str, Any]:
    """Merge each project's AOI into one FeatureCollection, tagged with `tm_project_id`."""
    features: list[dict[str, Any]] = []
    for proj in projects:
        raw = proj.aoi
        raw_features = (
            raw.get("features", [])
            if raw.get("type") == "FeatureCollection"
            else [{"type": "Feature", "geometry": raw, "properties": {}}]
        )
        for f in raw_features:
            raw_props = f.get("properties")
            props = dict(raw_props) if isinstance(raw_props, dict) else {}
            props["tm_project_id"] = proj.project_id
            features.append({"type": "Feature", "geometry": f["geometry"], "properties": props})
    return {"type": "FeatureCollection", "features": features}
