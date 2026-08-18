import torch

from semicon_restore.losses import (
    RestorationLoss,
    focal_frequency_loss,
    haar_highband_charbonnier,
    multiscale_ssim,
)
from semicon_restore.models import ModelConfig, build_model


def tiny_model():
    return build_model(ModelConfig(width=8, blocks=(1, 1, 1), conditioning=True))


def test_dynamic_two_times_shape():
    model = tiny_model()
    for shape in ((16, 16), (20, 24)):
        output = model(torch.randn(2, 1, *shape))
        assert output.shape == (2, 1, shape[0] * 2, shape[1] * 2)


def test_finite_backward():
    model = tiny_model()
    source = torch.randn(1, 1, 16, 16)
    target = torch.rand(1, 1, 32, 32)
    output = model(source)
    loss, _ = RestorationLoss()(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_input_features_preserve_raw():
    raw = torch.tensor([[[[-0.2, 1.4]]]])
    features = tiny_model().input_features(raw)
    assert torch.equal(features[:, :1], raw)
    assert features.shape[1] == 4


def test_haar_highband_loss_responds_to_detail():
    target = torch.zeros(1, 1, 8, 8)
    prediction = target.clone()
    prediction[..., ::2, ::2] = 1.0
    matching = haar_highband_charbonnier(target, target)
    different = haar_highband_charbonnier(prediction, target)
    assert different > matching


def test_frequency_weight_has_finite_backward():
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    target = torch.rand_like(prediction)
    loss, parts = RestorationLoss(frequency_weight=0.03)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["frequency"] > 0
    assert torch.isfinite(prediction.grad).all()


def test_context_model_shape_and_backward():
    config = ModelConfig(name="context_naf", width=8, blocks=(1, 1, 1), kernel_size=7,
                         intro_kernel_size=5, bottleneck_attention_blocks=1, attention_heads=4)
    model = build_model(config)
    source = torch.randn(1, 1, 16, 20)
    output = model(source)
    output.mean().backward()
    assert output.shape == (1, 1, 32, 40)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_multiscale_ssim_identical_is_one():
    image = torch.rand(2, 1, 32, 32)
    assert torch.allclose(multiscale_ssim(image, image), torch.tensor(1.0), atol=1e-4)


def test_focal_frequency_loss_detects_frequency_error():
    target = torch.zeros(1, 1, 16, 16)
    prediction = target.clone()
    prediction[..., ::2, ::2] = 1.0
    assert focal_frequency_loss(target, target) == 0
    assert focal_frequency_loss(prediction, target) > 0


def test_context_loss_is_finite_for_flat_and_sparse_targets():
    loss_function = RestorationLoss(0.75, 0.10, 0.10, 0.05, "ms_ssim", 1.0, "focal")
    for target in (torch.zeros(2, 1, 64, 64), torch.ones(2, 1, 64, 64)):
        target[0, 0, 32, 32] = 1 - target[0, 0, 32, 32]
        prediction = torch.randn_like(target, requires_grad=True)
        loss, parts = loss_function(prediction, target)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in parts.values())
        assert torch.isfinite(prediction.grad).all()
