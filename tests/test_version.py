from kalinov_bridge import __version__


def test_version_is_semver_zero_dot_something() -> None:
    assert __version__.startswith("0.")
