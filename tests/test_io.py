from pathlib import Path

import numpy as np
import pytest

from semicon_restore.io import (
    ArrayContractError,
    discover_npy,
    load_array,
    save_array_atomic,
)


def test_npy_contract(tmp_path: Path):
    source = tmp_path / "input.npy"
    np.save(source, np.zeros((8, 8), dtype=np.float32))
    assert discover_npy(tmp_path) == [source]
    assert load_array(source).shape == (8, 8)


def test_rejects_nonfinite(tmp_path: Path):
    source = tmp_path / "bad.npy"
    np.save(source, np.array([[np.nan]], dtype=np.float32))
    with pytest.raises(ArrayContractError):
        load_array(source)


def test_atomic_output_contract(tmp_path: Path):
    target = tmp_path / "output.npy"
    save_array_atomic(target, np.ones((4, 4), dtype=np.float32))
    restored = np.load(target, allow_pickle=False)
    assert restored.dtype == np.float32
    assert restored.shape == (4, 4)
