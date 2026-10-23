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


def test_predict_request_without_a_pre_image_is_rejected(serve, monkeypatch):
    monkeypatch.setattr(
        serve, "_predict", lambda *args: pytest.fail("_predict reached without pre_image_uri")
    )
    server = serve.Server(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    body = json.dumps({"image_uri": "https://tiles.example/{z}/{x}/{y}", "bbox": [0, 0, 1, 1], "zoom": 18})
    request = urllib.request.Request(
        f"http://127.0.0.1:{server.server_address[1]}/predict", data=body.encode(), method="POST"
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=10)
    finally:
        server.shutdown()
        server.server_close()
    assert error.value.code == 400
    assert "pre_image_uri" in json.loads(error.value.read())["error"]


def test_predict_marks_buildings_over_a_pre_image_gap_as_no_data(serve, tmp_path, monkeypatch):
    rasters = _write_pair(tmp_path, blank_pre_columns=slice(0, 40))
    BUILDINGS.to_file(tmp_path / "buildings.geojson")
    monkeypatch.setattr(serve, "_download_raster", lambda tms, *args: str(rasters[tms]))
    monkeypatch.setattr(serve, "_fetch_osm_buildings", lambda *args: str(tmp_path / "buildings.geojson"))

    result = serve._predict("post", "pre", [0, 0, 1, 1], 18, 0.0)

    classes = {
        feature["properties"]["osm_id"]: feature["properties"]["damage_class"]
        for feature in result["features"]
    }
    assert classes[1] == NO_DATA_CLASS
    assert classes[2] != NO_DATA_CLASS
