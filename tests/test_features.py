import pytest
import torch

from semicon_restore.checkpoint import widen_input_weights
from semicon_restore.features import NoiseFeatures, input_channels
from semicon_restore.models import ModelConfig, build_model


def tiny(input_mode: str) -> ModelConfig:
    return ModelConfig(width=8, blocks=(1, 1, 1), input_mode=input_mode,
                       in_channels=input_channels(input_mode))


def tile(values: list[list[float]], size: int = 8) -> torch.Tensor:
    # Reflect padding requires the pad to be smaller than the axis it pads, so the noise-aware blur at
    # radius 2 needs at least 3 pixels a side, as does the model's own pad to a multiple of 4. Tiling to 8
    # keeps every test input above that floor while preserving the exact values a case is built around.
    pattern = torch.tensor(values, dtype=torch.float32)
    repeats = (size // pattern.shape[0], size // pattern.shape[1])
    return pattern.repeat(repeats)[None, None]


def test_input_channels_rejects_unknown_modes():
    with pytest.raises(ValueError):
        input_channels("raw6")


def test_model_config_requires_channels_to_match_the_input_mode():
    with pytest.raises(ValueError):
        ModelConfig(input_mode="noise_aware", in_channels=4)
    assert ModelConfig.from_dict({"input_mode": "noise_aware"}).in_channels == 8


def test_noise_aware_features_extend_rather_than_replace_the_raw_channels():
    # The first four channels have to stay bit-identical to raw4, because that is what lets a widened
    # stem reproduce the baseline function exactly.
    raw = tile([[-0.2, 0.4], [0.9, 1.4]])
    plain = NoiseFeatures("raw4")(raw)
    extended = NoiseFeatures("noise_aware")(raw)
    assert plain.shape[1] == 4
    assert extended.shape[1] == 8
    assert torch.equal(extended[:, :4], plain)


def test_noise_aware_features_stay_finite_on_out_of_range_inputs():
    # Adjacent extremes rather than a smooth gradient: this is the harshest case for the local-spread
    # channel, whose numerator is the observed standard deviation inside the blur window.
    features = NoiseFeatures("noise_aware")(tile([[-40.0, 0.0], [1.0, 60.0]]))
    assert torch.isfinite(features).all()
    # The whitened residual and the local-spread ratio are the two unbounded channels, so both carry an
    # explicit limit; a clipping artefact must not become an arbitrarily large activation.
    assert features[:, 5].abs().max() <= 6.0
    assert features[:, 7].max() <= 6.0


def test_variance_stabilized_channel_is_normalized_at_full_scale():
    features = NoiseFeatures("noise_aware")(torch.ones(1, 1, 8, 8))
    assert features[0, 6].mean().item() == pytest.approx(1.0)


def test_widening_the_stem_preserves_the_function_exactly():
    torch.manual_seed(2026)
    narrow = build_model(tiny("raw4")).eval()
    wide = build_model(tiny("noise_aware")).eval()
    state, notes = widen_input_weights(narrow.state_dict(), wide)
    wide.load_state_dict(state, strict=True)
    raw = torch.rand(2, 1, 16, 16) * 1.4 - 0.2
    with torch.inference_mode():
        assert torch.equal(narrow(raw), wide(raw))
    assert notes


def test_widening_leaves_an_unchanged_input_mode_untouched():
    torch.manual_seed(7)
    model = build_model(tiny("raw4"))
    state, notes = widen_input_weights(model.state_dict(), build_model(tiny("raw4")))
    assert not notes
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())


def test_noise_aware_model_trains_end_to_end():
    model = build_model(tiny("noise_aware"))
    output = model(torch.rand(2, 1, 16, 20))
    output.mean().backward()
    assert output.shape == (2, 1, 32, 40)
    assert model.intro.weight.grad is not None
    # A zero gradient on the added channels would mean the extra features never reach the network.
    assert model.intro.weight.grad[:, 4:].abs().sum() > 0
