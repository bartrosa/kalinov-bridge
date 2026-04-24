from pathlib import Path

import pytest

from kalinov_bridge.repo import find_repo_root


def test_find_repo_root_from_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('name = "kalinov-bridge"\n', encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    found = find_repo_root(nested)
    assert found == tmp_path


def test_find_repo_root_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path)
