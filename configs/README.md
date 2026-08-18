# Configurations

Versioned YAML for local training, Kaggle training, evaluation, and controlled ablations. Every run stores its fully resolved configuration in its checkpoint and report directory.

## Layering

A file may name its base with `extends`, resolved relative to the file that names it, and bases may extend bases in turn. `--base-config` on the command line sits below the whole chain. Nearest layer wins, merged key by key into nested sections, so an overlay restates only what it changes. One consequence to keep in mind: the merge is per key, not per section, so a change of `input_mode` must restate `in_channels` alongside it.

## Files

| File | Purpose |
| --- | --- |
| `train.yaml` | The full baseline. Self-contained; everything else layers on it. |
| `train-v2.yaml` | The v2 recipe: noise-aware 8-channel input, 50% synthetic pairs, ramped frequency term, final squared-error phase. Warm-start it with `--initialize kaggle-output/best.pt`. |
| `kaggle-v2.yaml` | Two-GPU Kaggle overlay on `train-v2`: paths, per-rank batch size, square-root-scaled learning rate, crop curriculum. Launch with `torchrun --nproc_per_node=2`. |
| `finetune-mse.yaml` | Short squared-error finetune from a converged checkpoint, when the remaining gap is metric rather than structure. |
| `finetune-detail-frequency.yaml` | Detail and frequency finetune. Partial overlay: needs `--base-config`. |
| `context-naf-from-scratch.yaml` | Context-conditioned variant, self-contained. |
| `context-naf-smoke.yaml`, `smoke.yaml`, `kaggle.yaml` | Partial overlays kept for the original single-base workflow: pass `--base-config configs/train.yaml`. |
| `degradation-v1.json` | Calibrated degradation parameters, produced by `scripts/calibrate_degradation.py` and read by both training and inference. |

`ablation/` holds the controlled arms; see the README there.
