from deeplob import __version__


def test_version_is_non_empty() -> None:
    assert __version__


def test_version_matches_expected_m1_tag() -> None:
    assert __version__ == "0.1.0-m1"
