# Tests

Run the whole suite from the repository root with `PYTHONPATH=src pytest tests/ -q`; nothing here needs the dataset or a checkpoint.

| File | Covers |
| --- | --- |
| `test_io.py` | NPY contract, atomic writes, output validation |
| `test_data.py` | Paired crop alignment, manifest reading, morphology-balanced sampling |
| `test_model.py` | Model shapes, finite gradients, mixed precision, checkpoint resume, inference portability |
| `test_degradation.py` | Calibrated first two moments of the synthetic degradation, Dirichlet detail coefficient, the variance-stabilising transform's flattening identity, parameter validation and disk round-trip |
| `test_features.py` | Noise-aware input channels extending `raw4` bit-identically, bounded channels under out-of-range input, exact function preservation when a stem is widened |
| `test_schedule.py` | Frequency-weight ramp, final squared-error phase, inertness without either, and the measured Charbonnier-to-MSE magnitude gap the phase weight has to absorb |
| `test_distributed.py` | `ShardSampler` partitioning without padding, summed shards reproducing the single-process mean, disabled `reduce_sum`, device wrapping |
| `test_inference.py` | Dihedral group properties, self-ensemble equivariance, checkpoint-ensemble averaging, back-projection limits and block-constant corrections |
| `test_config.py` | `extends` chains resolved relative to the referring file, circular-chain detection, `--base-config` layering |

Two conventions worth keeping. Tensor inputs are at least 3 pixels a side, because both `NoiseFeatures`' blur and the model's own pad to a multiple of four use reflect padding, which requires the pad to be smaller than the axis it pads. And the statistical tests in `test_degradation.py` use a 2x2 checkerboard rather than random data, so the block mean and within-block variance are exact by construction and the assertions test the moment identities instead of a tolerance band.
