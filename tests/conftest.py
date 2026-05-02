"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

# Mining tests use fixtures only; accidental outbound HTTP should fail fast in CI.
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:1")

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
