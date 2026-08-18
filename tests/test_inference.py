import pytest
import torch
from torch import nn

from semicon_restore.degradation import DEFAULT_PARAMS, area_downsample, block_repeat
from semicon_restore.inference import (
    BackProjection,
    Ensemble,
    SelfEnsemble,
    back_project,
    dihedral,
    dihedral_inverse,
)


class Upsample(nn.Module):
    # Nearest-neighbour 2x upsampling commutes with every rotation and transpose of the square, so this
    # stands in for the real model wherever a test needs a known-equivariant reference.
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return block_repeat(raw)


class Constant(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return torch.full((raw.shape[0], 1, raw.shape[2] * 2, raw.shape[3] * 2), self.value)


def test_dihedral_group_has_eight_distinct_elements():
    # A non-square input with distinct values is what separates the eight symmetries: on a square, or on
    # anything with a symmetry of its own, several of them would coincide and a wrong table would pass.
    x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    images = [dihedral(x, index) for index in range(8)]
    assert len({tuple(image.flatten().tolist()) for image in images}) == 8
    assert all(image.shape[-2:] == ((4, 6) if index & 4 == 0 and index % 2 == 0 else image.shape[-2:])
               for index, image in enumerate(images))


def test_every_dihedral_transform_is_exactly_invertible():
    x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    for index in range(8):
        assert torch.equal(dihedral_inverse(dihedral(x, index), index), x)


def test_four_way_mode_is_the_first_half_of_the_eight_way_one():
    x = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
    assert all(torch.equal(dihedral(x, index), torch.rot90(x, index, dims=(-2, -1))) for index in range(4))


def test_self_ensemble_leaves_an_equivariant_model_untouched():
    # The averaged prediction of an exactly equivariant model is the model's own prediction, so any
    # mismatch between a transform and its inverse shows up here as a difference from a plain forward
    # pass. Bit-exactness is achievable because every term of the average is identical.
    raw = torch.rand(2, 1, 8, 12)
    model = Upsample()
    for transforms in (1, 4, 8):
        assert torch.allclose(SelfEnsemble(model, transforms)(raw), model(raw), atol=1e-6)


def test_self_ensemble_averages_orientation_dependent_predictions():
    # A model that reads one fixed corner is maximally orientation dependent: each transform sends a
    # different pixel there, so the average is the mean over the four corners rather than any one of them.
    class Corner(nn.Module):
        def forward(self, raw: torch.Tensor) -> torch.Tensor:
            return raw[..., :1, :1].expand(-1, -1, raw.shape[2] * 2, raw.shape[3] * 2).clone()

    raw = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    averaged = SelfEnsemble(Corner(), 4)(raw)
    assert averaged.mean().item() == pytest.approx((1.0 + 2.0 + 3.0 + 4.0) / 4)


def test_ensemble_averages_its_members():
    raw = torch.rand(1, 1, 4, 4)
    assert Ensemble([Constant(1.0), Constant(3.0)])(raw).unique().tolist() == [2.0]
    assert torch.equal(Ensemble([Constant(2.0)])(raw), Constant(2.0)(raw))


def test_wrappers_reject_degenerate_configurations():
    with pytest.raises(ValueError):
        SelfEnsemble(Upsample(), 2)
    with pytest.raises(ValueError):
        Ensemble([])
    with pytest.raises(ValueError):
        BackProjection(Upsample(), DEFAULT_PARAMS, 0.0)


def test_zero_variance_back_projection_is_the_identity():
    prediction, raw = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 4, 4)
    assert torch.allclose(back_project(prediction, raw, DEFAULT_PARAMS, 1e-12), prediction, atol=1e-6)


def test_large_variance_back_projection_enforces_the_observation():
    # As the assumed model error grows the Wiener gain approaches one, and a gain of one makes the 2x2
    # average of the output equal the observation exactly. That limit is what "data consistency" means
    # here, so it is the property worth pinning rather than any particular intermediate gain.
    prediction, raw = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 4, 4)
    refined = back_project(prediction, raw, DEFAULT_PARAMS, 1e6)
    assert torch.allclose(area_downsample(refined), raw, atol=1e-4)


def test_back_projection_moves_every_pixel_of_a_block_together():
    # One observed value constrains a block mean and says nothing about the arrangement inside the block,
    # so the correction has to be constant across each 2x2 group. A per-pixel correction would be
    # inventing detail the measurement does not contain.
    prediction, raw = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 4, 4)
    correction = back_project(prediction, raw, DEFAULT_PARAMS, 1e-3) - prediction
    assert torch.allclose(correction, block_repeat(area_downsample(correction)), atol=1e-6)


def test_back_projection_gain_shrinks_where_the_signal_is_noisier():
    # The gain is variance / (variance + noise_variance), and the calibrated noise grows with intensity,
    # so a bright block must be corrected less than a dark one from the same residual.
    raw = torch.tensor([[[[0.05, 0.95]]]])
    prediction = torch.zeros(1, 1, 2, 4)
    correction = back_project(prediction, raw, DEFAULT_PARAMS, 1e-3) - prediction
    assert correction[0, 0, 0, 0] / 0.05 > correction[0, 0, 0, 2] / 0.95


def test_back_projection_module_matches_the_function_it_wraps():
    raw = torch.rand(1, 1, 4, 4)
    model = Upsample()
    wrapped = BackProjection(model, DEFAULT_PARAMS, 1e-3)(raw)
    assert torch.allclose(wrapped, back_project(model(raw), raw, DEFAULT_PARAMS, 1e-3), atol=1e-6)
