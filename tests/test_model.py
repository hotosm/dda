import pytest
import torch
from huggingface_hub import hf_hub_download
from torchmetrics.classification import MulticlassF1Score

from dda.config import N_DAMAGE_CLASSES
from dda.model import DinoV3DamageNet, _scored_classes, damage_scores

LABELLED = [0] * 90 + [3] * 10


def _scores(predicted: list[int], labelled: list[int]) -> dict[str, torch.Tensor]:
    f1 = MulticlassF1Score(num_classes=N_DAMAGE_CLASSES, average=None)
    f1.update(torch.tensor(predicted), torch.tensor(labelled))  # ty: ignore[invalid-argument-type]
    return damage_scores(f1.compute(), _scored_classes(f1))  # ty: ignore[missing-argument]


def test_damage_scores_average_only_the_classes_labelled_or_predicted():
    scores = _scores(LABELLED, LABELLED)
    assert scores["dmg_macro_f1"] == pytest.approx(1.0)
    assert scores["dmg_f1"] == pytest.approx(1.0, abs=1e-5)
    assert scores["destroyed_f1"] == pytest.approx(1.0)


def test_damage_scores_count_a_predicted_class_the_labels_never_contain():
    scores = _scores([1, *LABELLED[1:]], LABELLED)
    assert scores["dmg_macro_f1"] == pytest.approx(((178 / 179) + 0.0 + 1.0) / 3)


def test_damage_scores_drop_when_destroyed_buildings_are_missed():
    scores = _scores([0] * 100, LABELLED)
    assert scores["destroyed_f1"] == 0.0
    assert scores["dmg_macro_f1"] == pytest.approx((180 / 190) / 2)


def test_damage_scores_reject_an_empty_set_of_scored_classes():
    with pytest.raises(ValueError, match="no labelled or predicted"):
        damage_scores(torch.zeros(N_DAMAGE_CLASSES), torch.zeros(N_DAMAGE_CLASSES, dtype=torch.bool))


@pytest.mark.slow
def test_forward_shapes_siamese_and_post_only():
    ckpt = hf_hub_download("kshitijrajsharma/dinov3", "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth")
    net = DinoV3DamageNet(
        ckpt_path=ckpt,
        seg_out_indices=(5, 11, 17, 23),
        decoder_channels=256,
        use_pre=True,
        modality_dropout=0.0,
    ).eval()
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in net.backbone.parameters())
    assert n_train < 40e6

    post = torch.randn(1, 3, 512, 512)
    pre = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        loc, dmg = net(post, pre)
    assert loc.shape == (1, 512, 512)
    assert dmg.shape == (1, 4, 512, 512)

    with torch.no_grad():
        _, dmg2 = net(post)  # pre omitted -> graceful post-only
    assert dmg2.shape == (1, 4, 512, 512)
