import importlib
import json
import sys
import threading
import urllib.error
import urllib.request

import pytest
import torch
from test_infer_onnx import BUILDINGS, _PairConv, _write_pair

from dda import infer
from dda.pool import NO_DATA_CLASS


@pytest.fixture(scope="module")
def serve():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DDA_DEVICE", "cpu")
        patch.delenv("DDA_DAMAGE_CKPT", raising=False)
        patch.setattr(infer, "resolve_ckpt", lambda *args, **kwargs: "unused.ckpt")
        torch.manual_seed(0)
        patch.setattr(infer, "load_model", lambda *args, **kwargs: _PairConv().eval())
        sys.modules.pop("dda.serve", None)
        yield importlib.import_module("dda.serve")
    sys.modules.pop("dda.serve", None)


POST = "https://post.example/{z}/{x}/{y}"
PRE = "https://pre.example/{z}/{x}/{y}"


def _post(serve, body: dict) -> tuple[int, dict]:
    server = serve.Server(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/predict", data=json.dumps(body).encode(), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    finally:
        server.shutdown()
        server.server_close()


def test_predict_request_passes_the_named_imagery_to_the_prediction(serve, monkeypatch):
    calls = []
    monkeypatch.setattr(
        serve, "_predict", lambda *args: calls.append(args) or {"type": "FeatureCollection", "features": []}
    )
    imagery = {"post": "https://post.example/{z}/{x}/{y}", "pre": "https://pre.example/{z}/{x}/{y}"}

    status, _ = _post(serve, {"imagery": imagery, "bbox": [0, 0, 1, 1], "zoom": 18})

    assert status == 200
    assert calls == [(imagery, [0.0, 0.0, 1.0, 1.0], 18, 0.0)]


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ({"imagery": {"post": POST}, "bbox": [0, 0, 1, 1], "zoom": 18}, "imagery requires"),
        ({"imagery": {"post": POST, "pre": 1}, "bbox": [0, 0, 1, 1], "zoom": 18}, "imagery requires"),
        ({"imagery": {"post": POST, "pre": PRE}}, "'bbox'"),
        ({"imagery": {"post": POST, "pre": PRE}, "bbox": [0, 0, 1], "zoom": 18}, "bbox needs 4 numbers"),
        ({"imagery": {"post": POST, "pre": PRE}, "bbox": [0, 0, 1, 1], "zoom": 18, "params": "x"}, "params"),
    ],
    ids=["missing pre", "pre not a url", "missing bbox", "bbox of 3", "params not an object"],
)
def test_predict_request_with_invalid_fields_is_rejected(serve, monkeypatch, body, reason):
    monkeypatch.setattr(
        serve, "_predict", lambda *args: pytest.fail("_predict reached with an invalid request")
    )
    status, response = _post(serve, body)
    assert status == 400
    assert response["error"].startswith("invalid request: ")
    assert reason in response["error"]


def test_predict_marks_buildings_over_a_pre_image_gap_as_no_data(serve, tmp_path, monkeypatch):
    rasters = _write_pair(tmp_path, blank_pre_columns=slice(0, 40))
    BUILDINGS.to_file(tmp_path / "buildings.geojson")
    monkeypatch.setattr(serve, "_download_raster", lambda tms, *args: str(rasters[tms]))
    monkeypatch.setattr(serve, "_fetch_osm_buildings", lambda *args: str(tmp_path / "buildings.geojson"))

    result = serve._predict({"post": "post", "pre": "pre"}, [0, 0, 1, 1], 18, 0.0)

    classes = {
        feature["properties"]["osm_id"]: feature["properties"]["damage_class"]
        for feature in result["features"]
    }
    assert classes[1] == NO_DATA_CLASS
    assert classes[2] != NO_DATA_CLASS
