from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from semicon_restore import distributed
from semicon_restore.config import load_config
from semicon_restore.data import (
    DEFAULT_SCALE_RANGE,
    PairedNpyDataset,
    morphology_balanced_sampler,
    pair_paths,
    read_manifest,
)
from semicon_restore.degradation import DEFAULT_PARAMS_PATH
from semicon_restore.engine import train
from semicon_restore.models import ModelConfig, build_model
from semicon_restore.runtime import (
    DeviceInfo,
    environment_info,
    select_device,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the semiconductor restoration model.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--resume", default=None, help="Resume from a versioned training checkpoint")
    parser.add_argument("--initialize", default=None, help="Initialize model and EMA weights without optimizer state")
    args = parser.parse_args()
    config = load_config(args.config, args.base_config)
    # The process group has to exist before the device is chosen, because joining it is what pins this
    # process to its own GPU; select_device then reads that pinned device instead of defaulting to zero.
    info = distributed.initialize(config.get("device", "auto"))
    set_seed(config["seed"])
    device_info = select_device(config.get("device", "auto"))
    if info.primary:
        print({**environment_info(), "selected_precision": device_info.precision,
               "memory_gb": device_info.memory_gb, "world_size": info.world_size})
    try:
        run(args, config, info, device_info)
    finally:
        distributed.shutdown()


def run(args: argparse.Namespace, config: dict, info: distributed.DistributedInfo,
        device_info: DeviceInfo) -> None:
    root = Path(config["data_root"])
    all_pairs = pair_paths(root / config["train_lr_dir"], root / config["train_gt_dir"])
    manifest = read_manifest(config["split_manifest"])
    train_names, val_names = set(manifest["train"]), set(manifest["validation"])
    train_pairs = [pair for pair in all_pairs if pair[0].name in train_names]
    val_pairs = [pair for pair in all_pairs if pair[0].name in val_names]
    settings = config["training"]
    if settings.get("max_train_samples"):
        train_pairs = train_pairs[: int(settings["max_train_samples"])]
    if settings.get("max_validation_samples"):
        val_pairs = val_pairs[: int(settings["max_validation_samples"])]
    train_data = PairedNpyDataset(
        train_pairs,
        settings["lr_crop_size"],
        True,
        config["seed"],
        float(settings.get("edge_crop_probability", 0.0)),
        int(settings.get("detail_crop_candidates", 8)),
        synthetic_probability=float(settings.get("synthetic_probability", 0.0)),
        degradation=config.get("degradation", str(DEFAULT_PARAMS_PATH)),
        scale_range=tuple(settings.get("scale_range", DEFAULT_SCALE_RANGE)),
    )
    if info.primary and train_data.synthetic_probability > 0:
        print(f"synthetic_pairs probability={train_data.synthetic_probability} "
              f"scale_range={train_data.scale_range} degradation={train_data.degradation.source}")
    val_data = PairedNpyDataset(val_pairs, None, False, config["seed"])
    sampler = None
    if settings.get("morphology_balanced_sampling", False):
        # Each rank draws its own share of the weighted samples from a rank-specific stream. Weighted
        # sampling draws with replacement anyway, so independent per-rank draws stay faithful to the
        # intended morphology distribution without the ranks having to coordinate.
        samples = int(settings.get("samples_per_epoch", len(train_pairs)))
        sampler = morphology_balanced_sampler(train_pairs, config["seed"] + info.rank,
                                              max(1, samples // info.world_size))
    elif info.enabled:
        sampler = DistributedSampler(train_data, num_replicas=info.world_size, rank=info.rank,
                                     shuffle=True, seed=int(config["seed"]))
    curriculum = bool(settings.get("crop_curriculum"))
    # Only the distributed path gets an explicit loader generator: it is what makes the per-worker seeds
    # differ between ranks, so augmentation and synthetic noise are not drawn identically everywhere.
    generator = torch.Generator().manual_seed(int(config["seed"]) + 1000 * info.rank) if info.enabled else None
    train_loader = DataLoader(train_data, batch_size=settings["batch_size"], shuffle=sampler is None,
                              sampler=sampler, generator=generator,
                              num_workers=settings["num_workers"], pin_memory=device_info.device.type == "cuda",
                              persistent_workers=settings["num_workers"] > 0 and not curriculum)
    val_loader = DataLoader(val_data, batch_size=int(settings.get("validation_batch_size", 1)), shuffle=False,
                            sampler=distributed.ShardSampler(len(val_data), info) if info.enabled else None,
                            num_workers=settings["num_workers"],
                            pin_memory=device_info.device.type == "cuda", persistent_workers=settings["num_workers"] > 0)
    model_config = ModelConfig.from_dict(config["model"])
    model = build_model(model_config).to(device_info.device)
    result = train(model, train_loader, val_loader, config, model_config.to_dict(), device_info.device,
                   device_info.precision, args.resume, args.initialize, info)
    if info.primary:
        print(result)


if __name__ == "__main__":
    main()
