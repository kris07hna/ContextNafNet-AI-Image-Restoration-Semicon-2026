from __future__ import annotations

import argparse

from semicon_restore.audit import write_json
from semicon_restore.splits import create_grouped_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a structure-grouped train/validation split.")
    parser.add_argument("--lr-dir", default="train/train/NoisyLR")
    parser.add_argument("--gt-dir", default="train/train/GT")
    parser.add_argument("--output", default="splits/grouped-v1.json")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.985)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    split = create_grouped_split(args.lr_dir, args.gt_dir, args.validation_fraction, args.threshold, args.seed)
    write_json(args.output, split)
    print(f"Wrote {len(split['train'])} train and {len(split['validation'])} validation files to {args.output}")


if __name__ == "__main__":
    main()
