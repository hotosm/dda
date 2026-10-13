from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box
from torch import nn

from dda.config import N_DAMAGE_CLASSES
from dda.infer import predict_damage_onnx, read_aligned_pair, sliding_window_prob, sliding_window_prob_onnx
from dda.pool import assign_damage


class _PairConv(nn.Module):
    """Per-pixel logits from post and pre with distinct weights, so swapped inputs change the output."""

    def __init__(self) -> None:
        super().__init__()
        self.post = nn.Conv2d(3, N_DAMAGE_CLASSES, kernel_size=3, padding=1)
        self.pre = nn.Conv2d(3, N_DAMAGE_CLASSES, kernel_size=3, padding=1)

    def forward(self, post: torch.Tensor, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dmg = self.post(post) - self.pre(pre)
        return dmg[:, :1], dmg


class _DamageLogits(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, post: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        return self.net(post, pre)[1]


def _export(net: nn.Module, window: int, path: str) -> ort.InferenceSession:
    # Distinct example tensors: the dynamo exporter treats one tensor passed twice as a single aliased input.
    post_sample = torch.zeros(1, 3, window, window)
    pre_sample = torch.zeros(1, 3, window, window)
    torch.onnx.export(
        _DamageLogits(net).eval(),
        (post_sample, pre_sample),
        path,
        input_names=["post", "pre"],
        output_names=["damage_logits"],
        dynamo=True,
    )
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def test_onnx_sliding_window_matches_torch_on_uneven_image(tmp_path):
    window, stride, temperature = 32, 24, 1.5
    rng = np.random.default_rng(0)
    post = rng.integers(0, 256, size=(70, 90, 3), dtype=np.uint8)
    pre = rng.integers(0, 256, size=(70, 90, 3), dtype=np.uint8)
    torch.manual_seed(0)
    net = _PairConv().eval()
    session = _export(net, window, str(tmp_path / "pair.onnx"))

    expected = sliding_window_prob(net, post, pre, window, stride, temperature, "cpu")  # ty: ignore[invalid-argument-type]
    actual = sliding_window_prob_onnx(session, post, pre, window, stride, temperature)

    assert actual.shape == (N_DAMAGE_CLASSES, 70, 90)
    np.testing.assert_allclose(actual.sum(axis=0), 1.0, atol=1e-5)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_predict_damage_onnx_matches_the_torch_path_per_building(tmp_path):
    rng = np.random.default_rng(3)
    transform = from_origin(500000, 3100000, 0.5, 0.5)
    rasters = {}
    for name in ("post", "pre"):
        rasters[name] = tmp_path / f"{name}.tif"
        with rasterio.open(
            rasters[name],
            "w",
            driver="GTiff",
            width=80,
            height=60,
            count=3,
            dtype="uint8",
            crs="EPSG:32645",
            transform=transform,
        ) as dst:
            dst.write(rng.integers(0, 256, size=(3, 60, 80), dtype=np.uint8))
    buildings = gpd.GeoDataFrame(
        {"osm_id": [1, 2]},
        geometry=[box(500005, 3099980, 500015, 3099990), box(500025, 3099975, 500035, 3099995)],
        crs="EPSG:32645",
    )
    cfg = SimpleNamespace(
        radiometric_normalize=True,
        tile_window=32,
        tile_stride=24,
        temperature=1.0,
        pool_op="percentile",
        pool_percentile=80.0,
    )
    torch.manual_seed(0)
    net = _PairConv().eval()
    session = _export(net, cfg.tile_window, str(tmp_path / "pair.onnx"))

    actual = predict_damage_onnx(session, rasters["post"], rasters["pre"], buildings, cfg)

    post, pre, grid, crs = read_aligned_pair(cfg, rasters["post"], rasters["pre"])
    prob = sliding_window_prob(net, post, pre, cfg.tile_window, cfg.tile_stride, cfg.temperature, "cpu")  # ty: ignore[invalid-argument-type]
    expected = assign_damage(prob, grid, crs, buildings, pool_op=cfg.pool_op, percentile=cfg.pool_percentile)
    assert list(actual["damage_class"]) == list(expected["damage_class"])
    np.testing.assert_allclose(actual["damage_confidence"], expected["damage_confidence"], atol=1e-5)
