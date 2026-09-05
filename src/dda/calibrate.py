"""Temperature scaling for calibrated per-building confidence; single scalar fit on val by NLL."""

import logging

import torch
from torch import nn

from dda.infer import load_model
from dda.losses import IGNORE_INDEX
from dda.train import make_datamodule

log = logging.getLogger(__name__)


def fit_temperature(cfg, ckpt_path: str, split: str = "val", device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, cfg, device=device)

    dm = make_datamodule(cfg, 100.0)
    dm.setup("validate" if split == "val" else "test")
    loader = dm.val_dataloader() if split == "val" else dm.test_dataloader()

    logits_all, targets_all = [], []
    with torch.inference_mode():
        for batch in loader:
            pre = batch["pre"].to(device) if "pre" in batch else None
            _, dmg = model(batch["post"].to(device), pre)
            damage = batch["damage"].to(device)
            valid = damage != IGNORE_INDEX
            if valid.any():
                flat = dmg.permute(0, 2, 3, 1)[valid]
                logits_all.append(flat.float().cpu())
                targets_all.append(damage[valid].cpu())

    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    log_t = nn.Parameter(torch.zeros(1))  # optimise log(T) to keep T positive
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50)
    nll = nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(_closure)
    temperature = float(log_t.exp())
    before = float(nll(logits, targets))
    after = float(nll(logits / log_t.exp(), targets))
    log.info("Fitted temperature=%.4f | val NLL %.4f -> %.4f", temperature, before, after)
    return temperature
