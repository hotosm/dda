"""Localization (binary) and ordinal damage losses; adjacent-class confusion costs less than far confusion."""

import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn
from torch.nn.functional import one_hot

IGNORE_INDEX = -100


class LocalizationLoss(nn.Module):
    """BCE + Dice on the binary building mask."""

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode="binary", from_logits=True)

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(logit, target) + self.dice(logit, target)


def _emd_ordinal(probs: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Squared Earth-Mover distance between predicted and one-hot CDFs so far-off classes cost more."""
    if valid.sum() == 0:
        return probs.sum() * 0.0
    n_classes = probs.shape[1]
    onehot = one_hot(target.clamp(min=0), num_classes=n_classes).to(probs.dtype)
    cdf_pred = torch.cumsum(probs, dim=1)
    cdf_true = torch.cumsum(onehot, dim=1)
    emd = ((cdf_pred - cdf_true) ** 2).sum(dim=1)
    return emd[valid].mean()


class OrdinalDamageLoss(nn.Module):
    """CE + squared-EMD over building pixels; background carries IGNORE_INDEX so the head skips it."""

    def __init__(self, class_weights: torch.Tensor | None = None, emd_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
        self.emd_weight = emd_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.shape[1]
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, n_classes)
        flat_target = target.reshape(-1)
        valid = flat_target != IGNORE_INDEX
        # Background-only crop: CE would be nan; contribute zero instead.
        if not bool(valid.any()):
            return logits.sum() * 0.0
        ce = self.ce(logits, target)
        probs = torch.softmax(flat_logits, dim=1)
        emd = _emd_ordinal(probs, flat_target, valid)
        return ce + self.emd_weight * emd
