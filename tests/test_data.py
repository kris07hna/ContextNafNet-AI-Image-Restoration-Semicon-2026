from pathlib import Path

import numpy as np

from semicon_restore.data import (
    PairedNpyDataset,
    _crop_pair,
    morphology_balanced_sampler,
    pair_paths,
)


def test_aligned_crop(tmp_path: Path):
    lr_dir, gt_dir = tmp_path / "lr", tmp_path / "gt"
    lr_dir.mkdir(); gt_dir.mkdir()
    lr = np.arange(64, dtype=np.float32).reshape(8, 8)
    gt = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    np.save(lr_dir / "a.npy", lr); np.save(gt_dir / "a.npy", gt)
    dataset = PairedNpyDataset(pair_paths(lr_dir, gt_dir), crop_size=4, training=False)
    sample = dataset[0]
    assert sample["lr"].shape == (1, 4, 4)
    assert sample["gt"].shape == (1, 8, 8)
    expected = sample["lr"].repeat_interleave(2, 1).repeat_interleave(2, 2)
    assert np.array_equal(sample["gt"].numpy(), expected.numpy())


def test_detail_crop_selects_stronger_gradient_candidate():
    class FixedRandom:
        def __init__(self):
            self.values = iter([0, 0, 4, 4])

        def random(self):
            return 0.0

        def randint(self, _lower, _upper):
            return next(self.values)

    lr = np.zeros((8, 8), dtype=np.float32)
    lr[4:, 4::2] = 1.0
    gt = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
    cropped_lr, cropped_gt = _crop_pair(lr, gt, 4, FixedRandom(), 1.0, 2)
    assert cropped_lr.mean() == 0.5
    assert np.array_equal(cropped_gt, np.repeat(np.repeat(cropped_lr, 2, axis=0), 2, axis=1))


def test_morphology_balanced_sampler_has_requested_length(tmp_path: Path):
    pairs = []
    for index in range(8):
        lr = np.zeros((8, 8), dtype=np.float32)
        lr[:, :: max(1, 8 - index)] = 1.0
        gt = np.repeat(np.repeat(lr, 2, axis=0), 2, axis=1)
        lr_path, gt_path = tmp_path / f"lr-{index}.npy", tmp_path / f"gt-{index}.npy"
        np.save(lr_path, lr); np.save(gt_path, gt)
        pairs.append((lr_path, gt_path))
    sampler = morphology_balanced_sampler(pairs, seed=1, samples_per_epoch=20)
    assert len(list(sampler)) == 20
