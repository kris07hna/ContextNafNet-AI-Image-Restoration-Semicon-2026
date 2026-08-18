import pytest
import yaml

from semicon_restore.config import load_config


def write(path, payload: dict) -> str:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_a_single_file_loads_unchanged(tmp_path):
    path = write(tmp_path / "one.yaml", {"lr": 0.1, "training": {"epochs": 3}})
    assert load_config(path) == {"lr": 0.1, "training": {"epochs": 3}}


def test_extends_merges_nested_sections_key_by_key(tmp_path):
    write(tmp_path / "base.yaml", {"data_root": "/data", "training": {"epochs": 100, "lr": 0.0002}})
    child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "training": {"epochs": 72}})
    config = load_config(child)
    assert config["training"] == {"epochs": 72, "lr": 0.0002}
    assert config["data_root"] == "/data"
    # The marker is a loader directive, not configuration, so it must not survive into the result and
    # reach the training script as an unknown key.
    assert "extends" not in config


def test_a_chain_of_overlays_resolves_nearest_first(tmp_path):
    # This is the kaggle-v2 -> train-v2 -> train layout. Without chaining, the middle layer would have to
    # restate every field of the bottom one to stay loadable on its own.
    write(tmp_path / "train.yaml", {"training": {"epochs": 100, "lr": 0.0002, "weight_decay": 0.01}})
    write(tmp_path / "recipe.yaml", {"extends": "train.yaml", "training": {"epochs": 100, "lr": 0.0002}})
    top = write(tmp_path / "top.yaml", {"extends": "recipe.yaml", "training": {"lr": 0.00028}})
    assert load_config(top)["training"] == {"epochs": 100, "lr": 0.00028, "weight_decay": 0.01}


def test_extends_resolves_against_the_referring_file_not_the_working_directory(tmp_path):
    # configs/ablation/*.yaml name ../train.yaml, so the chain has to be resolved relative to the file
    # that names it or the ablations would only load from one directory.
    write(tmp_path / "train.yaml", {"training": {"epochs": 100}})
    nested = tmp_path / "ablation"
    nested.mkdir()
    write(nested / "baseline.yaml", {"extends": "../train.yaml", "training": {"epochs": 30}})
    leaf = write(nested / "arm.yaml", {"extends": "baseline.yaml", "model": {"input_mode": "noise_aware"}})
    config = load_config(leaf)
    assert config["training"]["epochs"] == 30
    assert config["model"]["input_mode"] == "noise_aware"


def test_absolute_extends_paths_are_used_as_given(tmp_path):
    base = write(tmp_path / "base.yaml", {"training": {"epochs": 5}})
    other = tmp_path / "elsewhere"
    other.mkdir()
    child = write(other / "child.yaml", {"extends": base, "training": {"lr": 0.1}})
    assert load_config(child)["training"] == {"epochs": 5, "lr": 0.1}


def test_explicit_base_config_sits_below_the_whole_chain(tmp_path):
    # --base-config is the lowest layer, so an overlay chain above it still wins on every key it names.
    full = write(tmp_path / "full.yaml", {"training": {"epochs": 100, "lr": 0.0002, "grad_clip": 1.0}})
    write(tmp_path / "mid.yaml", {"extends": "full.yaml", "training": {"epochs": 20}})
    top = write(tmp_path / "smoke.yaml", {"extends": "mid.yaml", "training": {"lr": 0.001}})
    config = load_config(top, base_path=full)
    assert config["training"] == {"epochs": 20, "lr": 0.001, "grad_clip": 1.0}


def test_a_circular_chain_is_reported_rather_than_looped(tmp_path):
    write(tmp_path / "a.yaml", {"extends": "b.yaml", "x": 1})
    write(tmp_path / "b.yaml", {"extends": "a.yaml", "x": 2})
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_a_file_extending_itself_is_circular(tmp_path):
    write(tmp_path / "self.yaml", {"extends": "self.yaml"})
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "self.yaml")


def test_an_empty_file_is_an_empty_layer(tmp_path):
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
    base = write(tmp_path / "base.yaml", {"training": {"epochs": 4}})
    assert load_config(tmp_path / "empty.yaml") == {}
    assert load_config(tmp_path / "empty.yaml", base_path=base) == {"training": {"epochs": 4}}


def test_a_null_override_replaces_rather_than_merges(tmp_path):
    # finetune-mse.yaml sets final_phase_weights: null to switch the in-run phase transition off, which
    # only works if an explicit null overrides the inherited list instead of being skipped as empty.
    write(tmp_path / "base.yaml", {"training": {"final_phase_weights": [20.0, 0.1, 0.05, 0.0]}})
    child = write(tmp_path / "child.yaml", {"extends": "base.yaml", "training": {"final_phase_weights": None}})
    assert load_config(child)["training"]["final_phase_weights"] is None


def test_loading_does_not_mutate_a_shared_base(tmp_path):
    write(tmp_path / "base.yaml", {"training": {"epochs": 100, "lr": 0.0002}})
    first = write(tmp_path / "first.yaml", {"extends": "base.yaml", "training": {"epochs": 30}})
    second = write(tmp_path / "second.yaml", {"extends": "base.yaml", "training": {"epochs": 72}})
    assert load_config(first)["training"]["epochs"] == 30
    assert load_config(second)["training"]["epochs"] == 72
    assert load_config(tmp_path / "base.yaml")["training"]["epochs"] == 100
