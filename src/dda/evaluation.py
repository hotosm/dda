"""Per-building damage F1 by pooling model probs inside each ground-truth mask component."""

import logging

import numpy as np
import torch
from scipy import ndimage
from torchmetrics.functional.classification import multiclass_f1_score

from dda.config import DAMAGE_CLASSES, N_DAMAGE_CLASSES
from dda.infer import load_model
from dda.losses import IGNORE_INDEX
from dda.train import make_datamodule

log = logging.getLogger(__name__)
_ORDINAL = np.arange(N_DAMAGE_CLASSES)


def _pool_component(pix: np.ndarray, pool_op: str, percentile: float) -> int:
    if pool_op == "mean":
        return int(pix.mean(axis=1).argmax())
    severity = (_ORDINAL[:, None] * pix).sum(axis=0)
    return int(np.clip(round(float(np.percentile(severity, percentile))), 0, N_DAMAGE_CLASSES - 1))


def object_level_eval(
    cfg,
    ckpt_path: str,
    split: str = "val",
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, cfg, device=device)

    dm = make_datamodule(cfg, 100.0)
    dm.setup("validate" if split == "val" else "test")
    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

    gt_objs: list[int] = []
    pred_objs: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            post = batch["post"].to(device)
            pre = batch["pre"].to(device) if "pre" in batch else None
            _, dmg_logits = model(post, pre)
            probs = torch.softmax(dmg_logits, dim=1).float().cpu().numpy()
            damage = batch["damage"].numpy()
            for b in range(probs.shape[0]):
                gt = damage[b]
                building = gt != IGNORE_INDEX
                labels, n = ndimage.label(building)
                for i in range(1, n + 1):
                    comp = labels == i
                    gt_cls = int(np.bincount(gt[comp].astype(int), minlength=N_DAMAGE_CLASSES).argmax())
                    pred_cls = _pool_component(probs[b][:, comp], cfg.pool_op, cfg.pool_percentile)
                    gt_objs.append(gt_cls)
                    pred_objs.append(pred_cls)

    g = torch.tensor(gt_objs)
    p = torch.tensor(pred_objs)
    per_class = multiclass_f1_score(p, g, num_classes=N_DAMAGE_CLASSES, average=None)
    macro = float(per_class.mean())
    harmonic = float(len(per_class) / (1.0 / (per_class + 1e-6)).sum())
    result = {
        "split": split,
        "n_buildings": len(gt_objs),
        "per_class_f1": {name: round(float(f), 4) for name, f in zip(DAMAGE_CLASSES, per_class, strict=True)},
        "macro_f1": round(macro, 4),
        "harmonic_f1": round(harmonic, 4),
    }
    log.info("Object-level eval: %s", result)
    return result
