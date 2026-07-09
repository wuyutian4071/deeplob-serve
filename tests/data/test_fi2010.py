from pathlib import Path

import numpy as np
import pytest

from deeplob.data.fi2010 import load_fi2010
from deeplob.data.lob import NUM_FEATURES
from deeplob.data.synthetic import generate_synthetic_lob


def _write_fi2010_format_fixture(path: Path, features: np.ndarray, extra_rows: int = 0) -> None:
    """Writes `features` ([N, NUM_FEATURES]) transposed to `path`, simulating FI-2010's own
    on-disk layout (rows=features, columns=time). `extra_rows` appends that many additional
    rows of dummy values below the 40 feature rows, simulating FI-2010's real files having
    more than 40 rows (handcrafted features + 5 label rows) that load_fi2010 must ignore.
    """
    transposed = features.T  # [NUM_FEATURES, N]
    if extra_rows > 0:
        dummy = np.zeros((extra_rows, transposed.shape[1]))
        transposed = np.vstack([transposed, dummy])
    np.savetxt(path, transposed)


def test_loads_and_transposes_a_well_formed_fixture_back_to_the_original_features(
    tmp_path: Path,
) -> None:
    features = generate_synthetic_lob(num_snapshots=30, seed=3)
    fixture_path = tmp_path / "fi2010_fixture.txt"
    _write_fi2010_format_fixture(fixture_path, features)

    loaded = load_fi2010(fixture_path)

    assert loaded.shape == (30, NUM_FEATURES)
    np.testing.assert_allclose(loaded, features)


def test_ignores_rows_beyond_the_first_40_such_as_fi2010s_own_label_rows(
    tmp_path: Path,
) -> None:
    features = generate_synthetic_lob(num_snapshots=15, seed=4)
    fixture_path = tmp_path / "fi2010_with_extra_rows.txt"
    # Simulates FI-2010's real ~149-row format (40 raw features + handcrafted features +
    # label rows) -- the loader must read only the first 40 and ignore the rest.
    _write_fi2010_format_fixture(fixture_path, features, extra_rows=109)

    loaded = load_fi2010(fixture_path)

    assert loaded.shape == (15, NUM_FEATURES)
    np.testing.assert_allclose(loaded, features)


def test_raises_file_not_found_with_a_helpful_message(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.txt"
    with pytest.raises(FileNotFoundError, match="manual download"):
        load_fi2010(missing_path)


def test_rejects_a_file_with_too_few_rows(tmp_path: Path) -> None:
    fixture_path = tmp_path / "too_few_rows.txt"
    # Only 10 rows -- nowhere near NUM_FEATURES=40, so this can't be a valid
    # features-x-time FI-2010-format file.
    np.savetxt(fixture_path, np.zeros((10, 5)))

    with pytest.raises(ValueError, match="expected a >= 40-row"):
        load_fi2010(fixture_path)
