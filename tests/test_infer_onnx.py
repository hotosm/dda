from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box
from torch import nn

from dda.config import DINOV3_MEAN, DINOV3_STD, N_DAMAGE_CLASSES
from dda.export import _DamageHeadOnly
from dda.infer import sliding_window_prob
from dda.onnx_infer import predict_damage_onnx, sliding_window_prob_onnx
from dda.pool import NO_DATA_CLASS, assign_damage
from dda.tiling import read_aligned_pair


class _PairConv(nn.Module):
    """Per-pixel logits from post and pre with distinct weights, so swapped inputs change the output."""

    def __init__(self) -> None:
        super().__init__()
        self.post = nn.Conv2d(3, N_DAMAGE_CLASSES, kernel_size=3, padding=1)
        self.pre = nn.Conv2d(3, N_DAMAGE_CLASSES, kernel_size=3, padding=1)

    def forward(self, post: torch.Tensor, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dmg = self.post(post) - self.pre(pre)
        return dmg[:, :1], dmg


def _export(net: nn.Module, window: int, path: str) -> ort.InferenceSession:
    # Distinct example tensors: the dynamo exporter treats one tensor passed twice as a single aliased input.
    post_sample = torch.zeros(1, 3, window, window)
    pre_sample = torch.zeros(1, 3, window, window)
    torch.onnx.export(
        _DamageHeadOnly(net).eval(),
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


CFG = SimpleNamespace(
    radiometric_normalize=True,
    tile_window=32,
    tile_stride=24,
    temperature=1.0,
    pool_op="percentile",
    pool_percentile=80.0,
)
BUILDINGS = gpd.GeoDataFrame(
    {"osm_id": [1, 2]},
    geometry=[box(500005, 3099980, 500015, 3099990), box(500025, 3099975, 500035, 3099995)],
    crs="EPSG:32645",
)


def _write_pair(tmp_path: Path, blank_pre_columns: slice = slice(0, 0)) -> dict[str, Path]:
    """Random post and pre 80x60 rasters on one UTM grid; `blank_pre_columns` of pre are zero."""
    rng = np.random.default_rng(3)
    transform = from_origin(500000, 3100000, 0.5, 0.5)
    rasters = {}
    for name in ("post", "pre"):
        pixels = rng.integers(1, 256, size=(3, 60, 80), dtype=np.uint8)
        if name == "pre":
            pixels[:, :, blank_pre_columns] = 0
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
            dst.write(pixels)
    return rasters


def test_predict_damage_onnx_matches_the_torch_path_per_building(tmp_path):
    rasters = _write_pair(tmp_path)
    torch.manual_seed(0)
    net = _PairConv().eval()
    session = _export(net, CFG.tile_window, str(tmp_path / "pair.onnx"))

    actual = predict_damage_onnx(session, rasters["post"], rasters["pre"], BUILDINGS, CFG)

    post, pre, grid, crs = read_aligned_pair(CFG, rasters["post"], rasters["pre"])
    prob = sliding_window_prob(net, post, pre, CFG.tile_window, CFG.tile_stride, CFG.temperature, "cpu")  # ty: ignore[invalid-argument-type]
    expected = assign_damage(prob, grid, crs, BUILDINGS, pool_op=CFG.pool_op, percentile=CFG.pool_percentile)
    assert list(actual["damage_class"]) == list(expected["damage_class"])
    np.testing.assert_allclose(actual["damage_confidence"], expected["damage_confidence"], atol=1e-5)


def test_predict_damage_onnx_marks_buildings_over_a_pre_image_gap_as_no_data(tmp_path):
    # Columns 0-39 cover x 500000-500020: all of building 1, none of building 2.
    rasters = _write_pair(tmp_path, blank_pre_columns=slice(0, 40))
    torch.manual_seed(0)
    session = _export(_PairConv().eval(), CFG.tile_window, str(tmp_path / "pair.onnx"))

    actual = predict_damage_onnx(session, rasters["post"], rasters["pre"], BUILDINGS, CFG)

    assert actual["damage"].iloc[0] == "no-data"
    assert actual["damage_class"].iloc[0] == NO_DATA_CLASS
    assert np.isnan(actual["damage_confidence"].iloc[0])
    assert actual["damage_class"].iloc[1] != NO_DATA_CLASS


class _Pointwise(nn.Module):
    """1x1 convolutions, so every pixel's logits are independent of the window it sits in."""

    def __init__(self) -> None:
        super().__init__()
        self.post = nn.Conv2d(3, N_DAMAGE_CLASSES, kernel_size=1)
        self.pre = nn.Conv2d(3, N_DAMAGE_CLASSES, kernel_size=1)

    def forward(self, post: torch.Tensor, pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dmg = self.post(post) - self.pre(pre)
        return dmg[:, :1], dmg


def test_sliding_window_of_a_pointwise_model_equals_one_full_image_pass(tmp_path):
    window, stride, temperature = 32, 24, 1.5
    rng = np.random.default_rng(5)
    post = rng.integers(0, 256, size=(70, 90, 3), dtype=np.uint8)
    pre = rng.integers(0, 256, size=(70, 90, 3), dtype=np.uint8)
    torch.manual_seed(0)
    net = _Pointwise().eval()
    session = _export(net, window, str(tmp_path / "pointwise.onnx"))

    actual = sliding_window_prob_onnx(session, post, pre, window, stride, temperature)

    mean = torch.tensor(DINOV3_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(DINOV3_STD).view(1, 3, 1, 1)

    def normalize(image: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255.0 - mean) / std

    with torch.inference_mode():
        logits = net(normalize(post), normalize(pre))[1][0]
    expected = torch.softmax(logits / temperature, dim=0).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)
