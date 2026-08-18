import numpy as np
import pytest
import torch

from semicon_restore.degradation import (
    DEFAULT_PARAMS,
    DegradationParams,
    area_downsample,
    block_repeat,
    block_statistics,
    block_values,
    degrade,
    synthesize_pair,
    variance_stabilize,
)


def checkerboard(mean: float, spread: float, size: int = 32) -> np.ndarray:
    # Every 2x2 block is [[m-t, m+t], [m+t, m-t]], so the block mean is exactly m and the within-block
    # variance is exactly t^2. Pooling over identical blocks makes the sampling error small enough to
    # test the moment identities rather than merely their order of magnitude.
    tile = np.array([[mean - spread, mean + spread], [mean + spread, mean - spread]])
    return np.tile(tile, (size // 2, size // 2))


def test_block_statistics_match_construction():
    image = checkerboard(0.5, 0.1)
    mean, detail = block_statistics(image, DEFAULT_PARAMS)
    assert np.allclose(mean, 0.5)
    assert np.allclose(detail, 0.01)


def test_dirichlet_alpha_reproduces_the_fitted_detail_coefficient():
    # A Dirichlet(alpha) convex combination of the four block pixels has variance s^2 / (4 alpha + 1),
    # and the later multiplicative draw inflates it by (1 + quadratic). Solving for alpha has to invert
    # exactly that, or the synthesized detail noise misses the calibrated coefficient.
    params = DEFAULT_PARAMS
    assert 4 * params.dirichlet_alpha + 1 == pytest.approx((1 + params.quadratic) / params.detail)


def test_degrade_matches_the_calibrated_first_two_moments():
    params = DEFAULT_PARAMS
    image = checkerboard(0.5, 0.1)
    generator = np.random.default_rng(2026)
    samples = np.stack([degrade(image, generator, params) for _ in range(200)]).astype(np.float64)
    expected = params.noise_variance(0.5, 0.01)
    assert samples.mean() == pytest.approx(0.5, abs=3e-3)
    assert samples.var() == pytest.approx(expected, rel=0.05)


def test_degrade_variance_tracks_both_signal_and_detail():
    params = DEFAULT_PARAMS
    generator = np.random.default_rng(7)
    for mean, spread in ((0.2, 0.05), (0.8, 0.05), (0.5, 0.2)):
        samples = np.stack([degrade(checkerboard(mean, spread), generator, params)
                            for _ in range(200)]).astype(np.float64)
        assert samples.var() == pytest.approx(params.noise_variance(mean, spread ** 2), rel=0.06)


def test_area_downsample_inverts_block_repeat():
    for value in (np.random.default_rng(0).random((4, 6)), torch.rand(2, 1, 4, 6)):
        assert np.allclose(np.asarray(area_downsample(block_repeat(value))), np.asarray(value))


def test_block_values_partition_the_image():
    image = np.arange(16, dtype=np.float64).reshape(4, 4)
    values = block_values(image)
    assert values.shape == (4, 2, 2)
    assert sorted(values.ravel().tolist()) == sorted(image.ravel().tolist())
    assert np.allclose(values.mean(axis=0), area_downsample(image))


def test_synthesize_pair_returns_aligned_shapes_and_bounded_ground_truth():
    source = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    lr, gt = synthesize_pair(source, np.random.default_rng(1), DEFAULT_PARAMS, 32)
    assert gt.shape == (32, 32)
    assert lr.shape == (16, 16)
    assert gt.min() >= 0.0 and gt.max() <= 1.0


def test_variance_stabilizer_flattens_the_noise_scale_exactly():
    # dT/dx = 1 / (normalizer * noise_std(x)) by construction, so the product below is the same constant
    # at every intensity. That identity is the whole point of the transform: it is what turns a 14.6x
    # spread in noise level into a flat one.
    params = DegradationParams(detail_mode="none")
    stabilizer = params.stabilizer()
    x = torch.linspace(-0.4, 2.5, 64, dtype=torch.float64, requires_grad=True)
    variance_stabilize(x, stabilizer).sum().backward()
    products = x.grad * torch.as_tensor(params.noise_std(x.detach().numpy()))
    assert torch.allclose(products, torch.full_like(products, 1.0 / stabilizer.normalizer))


def test_variance_stabilizer_is_normalized_and_monotone_over_the_clip_range():
    stabilizer = DEFAULT_PARAMS.stabilizer()
    values = variance_stabilize(torch.linspace(DEFAULT_PARAMS.clip_low, DEFAULT_PARAMS.clip_high, 256), stabilizer)
    assert torch.isfinite(values).all()
    assert (values.diff() > 0).all()
    assert variance_stabilize(torch.ones(1), stabilizer).item() == pytest.approx(1.0)


def test_variance_stabilizer_survives_a_linear_term_too_small_for_the_closed_form():
    # 4ac > b^2 is what keeps T defined on all of R. A fit that violates it must inflate c rather than
    # produce a transform with a complex denominator over part of the input range.
    params = DegradationParams(quadratic=0.02, linear=0.05, constant=1e-6, detail_mode="none")
    values = variance_stabilize(torch.linspace(-0.5, 3.0, 256), params.stabilizer())
    assert torch.isfinite(values).all()
    assert (values.diff() > 0).all()


def test_degradation_params_reject_inconsistent_settings():
    with pytest.raises(ValueError):
        DegradationParams(quadratic=0.0)
    with pytest.raises(ValueError):
        DegradationParams(detail=1.5, detail_mode="blockmix")
    with pytest.raises(ValueError):
        DegradationParams(detail_mode="unknown")
    with pytest.raises(ValueError):
        DegradationParams(vst_margin=0.0)


def test_degradation_params_round_trip_through_disk(tmp_path):
    path = tmp_path / "degradation.json"
    DEFAULT_PARAMS.save(path, {"diagnostics": {"images": 3}})
    assert DegradationParams.load(path).to_dict() == DEFAULT_PARAMS.to_dict()
