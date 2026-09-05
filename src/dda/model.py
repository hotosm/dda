import logging
from collections.abc import Sequence
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from lightning.pytorch import LightningModule
from terratorch.models.decoders.upernet_decoder import UperNetDecoder
from terratorch.models.necks import LearnedInterpolateToPyramidal
from terratorch.registry import BACKBONE_REGISTRY
from torch import nn
from torchmetrics.classification import BinaryJaccardIndex, MulticlassF1Score

from dda.config import N_DAMAGE_CLASSES
from dda.losses import IGNORE_INDEX, LocalizationLoss, OrdinalDamageLoss

log = logging.getLogger(__name__)


def _download_ckpt(repo: str, filename: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, filename=filename))


def _harmonic_mean(values: torch.Tensor) -> torch.Tensor:
    """xView2 damage F1: harmonic mean over the per-class F1s, so one floored class collapses it."""
    return values.numel() / (1.0 / (values + 1e-6)).sum()


class DinoV3DamageNet(nn.Module):
    """Frozen siamese DINOv3 + UperNet; modality dropout swaps pre for post so one ckpt serves both."""

    def __init__(
        self,
        ckpt_path: str | Path | None,
        seg_out_indices: Sequence[int],
        decoder_channels: int,
        use_pre: bool,
        modality_dropout: float,
        unfreeze_last_n: int = 0,
        backbone_key: str = "terratorch_dinov3_vitl16",
    ):
        super().__init__()
        build_kwargs = {"ckpt_path": str(ckpt_path)} if ckpt_path else {}
        wrapper = BACKBONE_REGISTRY.build(backbone_key, **build_kwargs)
        self.backbone = wrapper.dinov3
        self.indices = list(seg_out_indices)
        self.use_pre = use_pre
        self.modality_dropout = modality_dropout
        embed_dim = self.backbone.embed_dim

        self.fusion = (
            nn.ModuleList(nn.Conv2d(2 * embed_dim, embed_dim, kernel_size=1) for _ in self.indices)
            if use_pre
            else None
        )
        channel_list = [embed_dim] * len(self.indices)
        self.pyramid = LearnedInterpolateToPyramidal(channel_list=channel_list)
        self.decoder = UperNetDecoder(
            embed_dim=list(self.pyramid.embedding_dim),
            channels=decoder_channels,
            pool_scales=(1, 2, 3, 6),
        )
        self.loc_head = self._head(decoder_channels, 1)
        self.dmg_head = self._head(decoder_channels, N_DAMAGE_CLASSES)

        self._freeze_backbone(unfreeze_last_n)

    @staticmethod
    def _head(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, out_ch, kernel_size=1),
        )

    def _freeze_backbone(self, unfreeze_last_n: int) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable_blocks: list[nn.Module] = []
        if unfreeze_last_n > 0:
            blocks = self.backbone.blocks
            for blk in list(blocks)[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad = True
                self.trainable_blocks.append(blk)
        self.backbone_trainable = unfreeze_last_n > 0

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        for blk in self.trainable_blocks:
            blk.train(mode)
        return self

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, dtype=torch.float16, enabled=x.is_cuda):
            if self.backbone_trainable:
                feats = self.backbone.get_intermediate_layers(
                    x, n=self.indices, reshape=True, norm=True, return_class_token=False
                )
            else:
                with torch.no_grad():
                    feats = self.backbone.get_intermediate_layers(
                        x, n=self.indices, reshape=True, norm=True, return_class_token=False
                    )
        return [f.float() for f in feats]

    def forward(
        self, post: torch.Tensor, pre: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = post.shape[-2:]
        post_feats = self._extract(post)

        if self.use_pre and self.fusion is not None:
            if pre is None:
                pre = post
            if self.training and self.modality_dropout > 0.0:
                drop = torch.rand(post.shape[0], device=post.device) < self.modality_dropout
                if drop.any():
                    pre = torch.where(drop.view(-1, 1, 1, 1), post, pre)
            pre_feats = self._extract(pre)
            fused = [
                conv(torch.cat([pf, qf], dim=1))
                for conv, pf, qf in zip(self.fusion, post_feats, pre_feats, strict=True)
            ]
        else:
            fused = post_feats

        pyramid_feats = self.pyramid(tuple(fused))
        decoded = self.decoder(pyramid_feats)
        upsample = torch.nn.functional.interpolate
        loc_logit = upsample(self.loc_head(decoded), size=(h, w), mode="bilinear", align_corners=False)
        dmg_logits = upsample(self.dmg_head(decoded), size=(h, w), mode="bilinear", align_corners=False)
        return loc_logit[:, 0], dmg_logits


class DinoV3DamageLit(LightningModule):
    def __init__(
        self,
        seg_out_indices: Sequence[int],
        decoder_channels: int,
        use_pre: bool,
        modality_dropout: float,
        lr: float,
        weight_decay: float,
        loc_loss_weight: float,
        dmg_loss_weight: float,
        class_weights: Sequence[float] | None = None,
        unfreeze_last_n: int = 0,
        onecycle_pct_start: float = 0.05,
        onecycle_div_factor: float = 25.0,
        onecycle_final_div_factor: float = 1e4,
        ckpt_path: str | Path | None = None,
        backbone_key: str = "terratorch_dinov3_vitl16",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ckpt_path"])
        self.net = DinoV3DamageNet(
            ckpt_path=ckpt_path,
            seg_out_indices=seg_out_indices,
            decoder_channels=decoder_channels,
            use_pre=use_pre,
            modality_dropout=modality_dropout,
            unfreeze_last_n=unfreeze_last_n,
            backbone_key=backbone_key,
        )
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.loc_loss = LocalizationLoss()
        self.dmg_loss = OrdinalDamageLoss(class_weights=weights)
        self.lr = lr
        self.weight_decay = weight_decay
        self.loc_loss_weight = loc_loss_weight
        self.dmg_loss_weight = dmg_loss_weight
        self.onecycle_pct_start = onecycle_pct_start
        self.onecycle_div_factor = onecycle_div_factor
        self.onecycle_final_div_factor = onecycle_final_div_factor
        self.val_loc_iou = BinaryJaccardIndex()
        self.val_dmg_f1 = MulticlassF1Score(num_classes=N_DAMAGE_CLASSES, average=None)
        self.test_loc_iou = BinaryJaccardIndex()
        self.test_dmg_f1 = MulticlassF1Score(num_classes=N_DAMAGE_CLASSES, average=None)

    def forward(self, post: torch.Tensor, pre: torch.Tensor | None = None):
        return self.net(post, pre)

    def _step(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = batch.get("pre")
        loc_logit, dmg_logits = self.net(batch["post"], pre)
        loc = self.loc_loss(loc_logit, batch["build_mask"].float())
        dmg = self.dmg_loss(dmg_logits, batch["damage"].long())
        loss = self.loc_loss_weight * loc + self.dmg_loss_weight * dmg
        return loss, loc_logit, dmg_logits

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loc_logit, dmg_logits = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        loc_pred = (torch.sigmoid(loc_logit) > 0.5).int()
        self.val_loc_iou.update(loc_pred, batch["build_mask"].int())  # ty: ignore[invalid-argument-type]
        damage = batch["damage"].long()
        valid = damage != IGNORE_INDEX
        if valid.any():
            preds = dmg_logits.argmax(dim=1)
            self.val_dmg_f1.update(preds[valid], damage[valid])  # ty: ignore[invalid-argument-type]

    def on_validation_epoch_end(self):
        per_class = self.val_dmg_f1.compute()  # ty: ignore[missing-argument]
        dmg_f1 = _harmonic_mean(per_class)
        macro_f1 = per_class.mean()
        loc_iou = self.val_loc_iou.compute()  # ty: ignore[missing-argument]
        # Macro is the checkpoint metric; harmonic collapses on any zero-count class in a single-event val.
        self.log("val/dmg_macro_f1", macro_f1, prog_bar=True)
        self.log("val/dmg_f1", dmg_f1, prog_bar=True)
        self.log("val/loc_iou", loc_iou, prog_bar=True)
        log.info(
            "epoch %d val/dmg_macro_f1=%.4f val/dmg_f1=%.4f loc_iou=%.4f per_class=%s",
            self.current_epoch,
            float(macro_f1),
            float(dmg_f1),
            float(loc_iou),
            [round(float(x), 3) for x in per_class],
        )
        self.val_dmg_f1.reset()
        self.val_loc_iou.reset()

    def test_step(self, batch, batch_idx):
        _, loc_logit, dmg_logits = self._step(batch)
        loc_pred = (torch.sigmoid(loc_logit) > 0.5).int()
        self.test_loc_iou.update(loc_pred, batch["build_mask"].int())  # ty: ignore[invalid-argument-type]
        damage = batch["damage"].long()
        valid = damage != IGNORE_INDEX
        if valid.any():
            preds = dmg_logits.argmax(dim=1)
            self.test_dmg_f1.update(preds[valid], damage[valid])  # ty: ignore[invalid-argument-type]

    def on_test_epoch_end(self):
        per_class = self.test_dmg_f1.compute()  # ty: ignore[missing-argument]
        self.log("test/dmg_macro_f1", per_class.mean())
        self.log("test/dmg_f1", _harmonic_mean(per_class))
        self.log("test/loc_iou", self.test_loc_iou.compute())  # ty: ignore[missing-argument]
        log.info("test per_class=%s", [round(float(x), 3) for x in per_class])
        self.test_dmg_f1.reset()
        self.test_loc_iou.reset()

    def configure_optimizers(self):
        params = [p for p in self.net.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=self.lr,
            total_steps=total_steps,
            pct_start=self.onecycle_pct_start,
            anneal_strategy="cos",
            div_factor=self.onecycle_div_factor,
            final_div_factor=self.onecycle_final_div_factor,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


def build_model(cfg) -> DinoV3DamageLit:
    ckpt_path = _download_ckpt(cfg.hf_ckpt_repo, cfg.hf_ckpt_file)
    return DinoV3DamageLit(
        ckpt_path=ckpt_path,
        seg_out_indices=tuple(cfg.seg_out_indices),
        decoder_channels=cfg.decoder_channels,
        use_pre=cfg.use_pre,
        modality_dropout=cfg.modality_dropout,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        loc_loss_weight=cfg.loc_loss_weight,
        dmg_loss_weight=cfg.dmg_loss_weight,
        class_weights=list(cfg.class_weights) if cfg.class_weights is not None else None,
        unfreeze_last_n=cfg.unfreeze_last_n,
        onecycle_pct_start=cfg.onecycle_pct_start,
        onecycle_div_factor=cfg.onecycle_div_factor,
        onecycle_final_div_factor=cfg.onecycle_final_div_factor,
        backbone_key=cfg.backbone,
    )
