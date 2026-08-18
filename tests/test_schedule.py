import pytest
import torch

from semicon_restore.engine import loss_schedule
from semicon_restore.losses import RestorationLoss

RAMP = {"frequency_start_fraction": 0.30, "frequency_ramp_fraction": 0.20}
FINAL = {"final_phase_fraction": 0.90, "final_phase_pixel_loss": "mse",
         "final_phase_weights": [20.0, 0.10, 0.05, 0.0]}


def criterion(**kwargs) -> RestorationLoss:
    return RestorationLoss(**{"frequency_weight": 0.03, **kwargs})


def test_frequency_term_is_absent_before_the_start_fraction():
    plan = loss_schedule(RAMP, criterion())
    assert plan(0.00)[0][3] == 0.0
    assert plan(0.29)[0][3] == 0.0
    # The first three weights are carried through untouched: the ramp adds a term, it does not
    # redistribute the ones already there.
    assert plan(0.00)[0][:3] == (0.70, 0.20, 0.10)


def test_frequency_term_ramps_linearly_to_the_configured_weight():
    plan = loss_schedule(RAMP, criterion())
    assert plan(0.35)[0][3] == pytest.approx(0.03 * 0.25)
    assert plan(0.40)[0][3] == pytest.approx(0.03 * 0.50)
    assert plan(0.50)[0][3] == pytest.approx(0.03)
    # Past the ramp the weight holds rather than continuing to grow.
    assert plan(0.99)[0][3] == pytest.approx(0.03)


def test_zero_ramp_switches_the_frequency_term_on_at_once():
    plan = loss_schedule({"frequency_start_fraction": 0.5, "frequency_ramp_fraction": 0.0}, criterion())
    assert plan(0.49)[0][3] == 0.0
    assert plan(0.50)[0][3] == pytest.approx(0.03)


def test_final_phase_replaces_both_the_weights_and_the_pixel_mode():
    plan = loss_schedule(RAMP | FINAL, criterion())
    assert plan(0.89) == ((0.70, 0.20, 0.10, pytest.approx(0.03)), "charbonnier")
    assert plan(0.90) == ((20.0, 0.10, 0.05, 0.0), "mse")
    assert plan(1.00) == ((20.0, 0.10, 0.05, 0.0), "mse")


def test_schedule_is_inert_without_a_final_phase_or_a_ramp():
    # configs/ablation/baseline.yaml sets neither, and every arm that adds one is compared against it, so
    # a plan that quietly drifted here would move all of the ablation results together.
    plan = loss_schedule({}, criterion(frequency_weight=0.0))
    assert {plan(progress) for progress in (0.0, 0.5, 0.9, 1.0)} == {((0.70, 0.20, 0.10, 0.0), "charbonnier")}


def test_null_final_phase_weights_leave_the_pixel_mode_alone():
    # finetune-mse.yaml runs squared error for the whole job and sets final_phase_weights: null, so the
    # in-run transition has to stay switched off rather than reappear with default weights.
    plan = loss_schedule({"final_phase_weights": None}, criterion(pixel_mode="mse", pixel_weight=20.0))
    assert plan(0.95) == ((20.0, 0.20, 0.10, pytest.approx(0.03)), "mse")


def test_schedule_rejects_malformed_final_phase_settings():
    with pytest.raises(ValueError):
        loss_schedule({"final_phase_weights": [20.0, 0.1, 0.05]}, criterion())
    with pytest.raises(ValueError):
        loss_schedule(FINAL | {"final_phase_pixel_loss": "l1"}, criterion())
    for fraction in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            loss_schedule(FINAL | {"final_phase_fraction": fraction}, criterion())


def test_restoration_loss_rejects_an_unknown_pixel_mode():
    with pytest.raises(ValueError):
        RestorationLoss(pixel_mode="l1")


def test_applying_a_plan_changes_which_pixel_loss_is_reported():
    # This is the assignment the training loop makes at engine.py:220. The point of the test is the
    # magnitude gap it exposes: the same error is reported ~39x smaller once squared, which is why the
    # final phase carries a pixel weight of 20 instead of the 0.70 it replaces.
    torch.manual_seed(2026)
    loss = criterion()
    target = torch.rand(2, 1, 32, 32)
    prediction = target + torch.randn_like(target) * 0.0164
    plan = loss_schedule(RAMP | FINAL, loss)
    loss.weights, loss.pixel_mode = plan(0.5)
    charbonnier = loss(prediction, target)[1]["pixel"].item()
    loss.weights, loss.pixel_mode = plan(0.95)
    mse = loss(prediction, target)[1]["pixel"].item()
    assert charbonnier / mse > 20
    # Weight times loss is what reaches the optimizer, and 20x squared error lands within a factor of two
    # of the 0.70x Charbonnier it replaces, so the final phase does not silently rescale the step size.
    assert 0.5 < (20.0 * mse) / (0.70 * charbonnier) < 2.0
