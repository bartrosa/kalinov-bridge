"""Unit tests for :mod:`kalinov.provers.lean.toolchain` helpers.

These tests do not require ``elan`` / ``lake`` / ``lean`` on PATH (they only
exercise pure path arithmetic in :func:`runtime_project_root`).
"""

from __future__ import annotations

from pathlib import Path

from kalinov.provers.lean.toolchain import runtime_project_root


def test_runtime_project_root_resolves_to_repo_runtime() -> None:
    """``runtime_project_root`` must point at ``<repo>/provers/lean/runtime``.

    Regression: an off-by-one in ``parents[N]`` previously resolved to
    ``<repo>/src/provers/lean/runtime``, which doesn't exist on disk and
    caused every Lean prover invocation (compile/check) to fail because
    ``lake`` would be launched from a missing cwd / outside the Lake project.
    """
    root = runtime_project_root()
    assert root.is_dir(), (
        f"runtime project root not found at {root!r}: "
        "this resolves relative to this source file's location and must point "
        "at the vendored Lake workspace next to the repo root."
    )
    assert (root / "lakefile.toml").is_file(), (
        f"expected a Lake project at {root!r} (missing lakefile.toml)"
    )
    assert root.name == "runtime"
    assert root.parent.name == "lean"
    assert root.parent.parent.name == "provers"
    # The directory immediately containing 'provers' must be the repo root,
    # never the package's own 'src' directory.
    assert root.parent.parent.parent.name != "src"


def test_runtime_project_root_is_outside_python_package() -> None:
    """The Lean runtime must not live inside the Python source tree.

    If it did, packaging would ship Lean sources to PyPI and the lake
    invocation would race against the installed wheel layout.
    """
    here = Path(__file__).resolve()
    repo_root = here.parent.parent
    src_dir = repo_root / "src"
    root = runtime_project_root()
    try:
        root.relative_to(src_dir)
    except ValueError:
        return
    raise AssertionError(
        f"runtime root {root!r} unexpectedly lives under the Python src/ tree",
    )
