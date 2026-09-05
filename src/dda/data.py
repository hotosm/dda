import logging

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.tv_tensors import Image, Mask

from dda.config import N_DAMAGE_CLASSES
from dda.losses import IGNORE_INDEX

log = logging.getLogger(__name__)

# DINOv3 was pretrained with ImageNet normalisation; a frozen backbone must see the same.
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)

# Field names in the EVER-Z/torchange_xView2 dataset: t1 = pre, t2 = post, t2_mask = damage raster
# coded 0=background, 1..4 = Joint Damage Scale (no-damage..destroyed).
PRE_FIELD = "t1_image"
POST_FIELD = "t2_image"
MASK_FIELD = "t2_mask"
NAME_FIELD = "image_name"


def _disaster(image_name: str) -> str:
    return image_name.rsplit("/", 1)[-1].rsplit("_", 1)[0]


def _build_transforms(img_size: int, train: bool) -> v2.Compose:
    """Geometry only; scaling and normalisation stay in the collate to share ops across pre+post."""
    if train:
        return v2.Compose(
            [v2.RandomCrop(img_size, pad_if_needed=True), v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()]
        )
    return v2.Compose([v2.CenterCrop(img_size)])


def _to_chw(pil_rgb) -> torch.Tensor:
    return torch.from_numpy(np.array(pil_rgb.convert("RGB"), copy=True)).permute(2, 0, 1)


class XbdDamageDataModule(LightningDataModule):
    """xBD pre/post tiles split by disaster event so val/test measure cross-event generalisation."""

    def __init__(
        self,
        repo_id: str,
        dataset_splits: tuple[str, ...],
        img_size: int,
        use_pre: bool,
        augment_photometric: bool,
        val_events: tuple[str, ...],
        test_events: tuple[str, ...],
        val_frac: float,
        data_pct: float,
        batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.dataset_splits = dataset_splits
        self.img_size = img_size
        self._normalize = v2.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD)
        self._photometric = (
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            if augment_photometric
            else None
        )
        self.use_pre = use_pre
        self.val_events = set(val_events)
        self.test_events = set(test_events)
        self.val_frac = val_frac
        self.data_pct = data_pct
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.seed = seed

    def _load_pool(self) -> Dataset:
        parts = [load_dataset(self.repo_id, split=s) for s in self.dataset_splits]
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    def _split(self, ds: Dataset, which: str) -> Dataset:
        held = self.val_events | self.test_events
        if which == "train":
            return ds.filter(lambda ex: _disaster(ex[NAME_FIELD]) not in held)
        events = self.val_events if which == "val" else self.test_events
        return ds.filter(lambda ex: _disaster(ex[NAME_FIELD]) in events)

    def _random_split(self, pool: Dataset) -> tuple[Dataset, Dataset]:
        """In-distribution split: proves the model learns damage, separate from cross-event."""
        shuffled = pool.shuffle(seed=self.seed)
        n_val = max(1, round(len(shuffled) * self.val_frac))
        return shuffled.select(range(n_val, len(shuffled))), shuffled.select(range(n_val))

    def setup(self, stage: str | None = None) -> None:
        self._train_tf = _build_transforms(self.img_size, train=True)
        self._eval_tf = _build_transforms(self.img_size, train=False)
        pool = self._load_pool()

        if self.val_events:
            train, val = self._split(pool, "train"), self._split(pool, "val")
            val_desc = f"events={sorted(self.val_events)}"
        else:
            train, val = self._random_split(pool)
            val_desc = f"random {self.val_frac:.0%}"

        if self.data_pct < 100.0:
            n = max(1, round(len(train) * self.data_pct / 100.0))
            train = train.shuffle(seed=self.seed).select(range(n))

        if stage in (None, "fit"):
            self.train_ds = train
            self.class_weights = self._estimate_class_weights(train)
            log.info("train tiles=%d", len(train))
        if stage in (None, "fit", "validate"):
            self.val_ds = val
            log.info("val tiles=%d (%s)", len(val), val_desc)
        if stage in (None, "test"):
            self.test_ds = self._split(pool, "test") if self.test_events else val

    def _estimate_class_weights(self, ds: Dataset, cap: int = 400) -> list[float]:
        counts = np.zeros(N_DAMAGE_CLASSES, dtype=np.float64)
        n = min(len(ds), cap)
        for i in range(n):
            raster = np.array(ds[i][MASK_FIELD])
            for code in range(1, N_DAMAGE_CLASSES + 1):
                counts[code - 1] += int((raster == code).sum())
        counts = np.maximum(counts, 1.0)
        weights = counts.sum() / (N_DAMAGE_CLASSES * counts)
        log.info("Class pixel counts (sample n=%d): %s", n, counts.tolist())
        log.info("Inverse-freq class weights: %s", weights.tolist())
        return weights.tolist()

    def _damage_target(self, raster: torch.Tensor) -> torch.Tensor:
        """Remap damage codes for buildings: 1..4 becomes 0..3, background code 0 becomes IGNORE_INDEX."""
        target = torch.full_like(raster, IGNORE_INDEX, dtype=torch.long)
        building = (raster >= 1) & (raster <= N_DAMAGE_CLASSES)
        target[building] = raster[building].long() - 1
        return target

    def _collate(self, examples: list[dict], train: bool) -> dict[str, torch.Tensor]:
        tf = self._train_tf if train else self._eval_tf
        posts, pres, masks, damages = [], [], [], []
        for ex in examples:
            post = _to_chw(ex[POST_FIELD])
            pre = _to_chw(ex[PRE_FIELD])
            raster = torch.from_numpy(np.array(ex[MASK_FIELD], copy=True).astype(np.int64)).unsqueeze(0)
            # Stack pre+post on the channel axis so the spatial transform is identical for both.
            stacked = Image(torch.cat([post, pre], dim=0))
            stacked_t, raster_t = tf(stacked, Mask(raster))
            stacked_t = stacked_t.float() / 255.0
            post_t, pre_t = stacked_t[:3], stacked_t[3:]
            if train and self._photometric is not None:
                post_t, pre_t = self._photometric(post_t), self._photometric(pre_t)
            posts.append(self._normalize(post_t))
            pres.append(self._normalize(pre_t))
            raster_v = raster_t.squeeze(0)
            masks.append((raster_v >= 1) & (raster_v <= N_DAMAGE_CLASSES))
            damages.append(self._damage_target(raster_v))
        batch = {
            "post": torch.stack(posts),
            "build_mask": torch.stack(masks).float(),
            "damage": torch.stack(damages),
        }
        if self.use_pre:
            batch["pre"] = torch.stack(pres)
        return batch

    def _loader(self, ds: Dataset, batch_size: int, shuffle: bool, train: bool) -> DataLoader:
        return DataLoader(
            ds,  # ty: ignore[invalid-argument-type]
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=shuffle,
            collate_fn=lambda b: self._collate(b, train=train),
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, self.batch_size, shuffle=True, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, self.eval_batch_size, shuffle=False, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, self.eval_batch_size, shuffle=False, train=False)
